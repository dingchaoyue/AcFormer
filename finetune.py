import os
import yaml
import time
import json
import random
import datetime
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from easydict import EasyDict
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from models.pivotal_train import RawMPA
from models.pivotal_train_modality import RawModalPA
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from optim import create_optimizer
from engines import raw_train, raw_evaluate
from scheduler import create_scheduler
from dataset import create_dataset, create_sampler, create_loader, msa_raw_collate_fn


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args, config):
    # ddp settings
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    config = EasyDict(config)

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    
    ######################################################################## Dataset #####################################################
    print("Creating Dataset")
    datasets = create_dataset(args.data, config)
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True,True,False], num_tasks, global_rank)         
    else:
        samplers = [None,None,None]
    collate_fn = [msa_raw_collate_fn, msa_raw_collate_fn, msa_raw_collate_fn]
    
    train_data_loader, val_data_loader, test_data_loader = create_loader(datasets, samplers, batch_size=[config['batch_size']]*3, 
                                num_workers=[4,4,4],
                                is_trains=[True,False,False], 
                                collate_fns=collate_fn)
    
    ######################################################################## Model #######################################################
    print("Creating Model")
    model = RawModalPA(config=config)
    model = model.to(device)   
    
    model_without_ddp = model
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    ######################################################################## Training #########################################################
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    
    print("Start training")
    start_time = time.time()
    best = 0
    best_epoch = 0

    for epoch in range(0, max_epoch):
        if args.distributed:
            train_data_loader.sampler.set_epoch(epoch)
        train_stats = raw_train(model, train_data_loader, optimizer, epoch, warmup_steps, device, lr_scheduler, config)           
        val_stats = raw_evaluate(model, val_data_loader, device, args, to_print=True)
        test_stats = raw_evaluate(model, test_data_loader, device, args, to_print=True)
        
        if utils.is_main_process():  
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                        }
            if float(val_stats['acc'])>best:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch)) 
                best = float(val_stats['acc'])
                best_epoch = epoch
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        lr_scheduler.step(epoch+warmup_steps+1)  
        dist.barrier()   

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
    if utils.is_main_process():   
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)               

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/test.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='./output/')

    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    
    # Mode
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--runs', type=int, default=5)

    # Bert
    parser.add_argument('--use_bert', type=str2bool, default=True)  # ! default True
    parser.add_argument('--use_cmd_sim', type=str2bool, default=True)

    # Train
    time_now = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    parser.add_argument('--name', type=str, default=f"{time_now}")
    parser.add_argument('--num_classes', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=10)
    parser.add_argument('--n_epoch', type=int, default=500)
    # parser.add_argument('--patience', type=int, default=6)  # ! default=6
    parser.add_argument('--patience', type=int, default=50)

    parser.add_argument('--diff_weight', type=float, default=0.3)
    parser.add_argument('--sim_weight', type=float, default=1.0)
    parser.add_argument('--sp_weight', type=float, default=0.0)
    parser.add_argument('--recon_weight', type=float, default=1.0)

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--clip', type=float, default=1.0)

    parser.add_argument('--rnncell', type=str, default='lstm')
    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--reverse_grad_weight', type=float, default=1.0)
    # Selectin activation from 'elu', "hardshrink", "hardtanh", "leakyrelu", "prelu", "relu", "rrelu", "tanh"
    parser.add_argument('--activation', type=str, default='relu')

    # model
    parser.add_argument('--model', type=str,
                        default='MISA', help='one of {MISA, }')

    # data
    parser.add_argument('--data', type=str, default='mosi_raw')

    # parser arguments
    args = parser.parse_args()
    print(args.data)

    if args.data == "mosi" or args.data == 'mosi_raw':
        args.num_classes = 1 
        args.batch_size = 64
    elif args.data == "mosei":
        args.num_classes = 1
        args.batch_size = 16
    elif args.data == "ur_funny":
        args.num_classes = 2
        args.batch_size = 32
    else:
        print("No dataset mentioned")
        exit()

    #  config definitations
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    

    
    main(args, config)