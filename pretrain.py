import argparse
import os
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from easydict import EasyDict

from models.pivotal_pretrain import AcFormer, AcFormerPretrain
from engines_pretrain import train, eval
# from models.tokenization_bert import BertTokenizer
from models.vit import interpolate_pos_embed
# from models.ast import interpolate_pos_embed

import utils
from dataset import create_dataset, create_sampler, create_loader, misa_pretrain_collate_fn
from scheduler import create_scheduler
from optim import create_optimizer
import logging
from timm.utils import setup_default_logging
import torch
from torch.utils.tensorboard import SummaryWriter


def get_logger():
    setup_default_logging(default_level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

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
    logger = get_logger()
    for key, value in config.items():
        logger.info("===>{0}:  {1}".format(key,value))

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    ######################################################################## Dataset ############################################################ 
    logger.info("Creating Dataset")
    datasets = [create_dataset(args.data, config)]
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True], num_tasks, global_rank)         
    else:
        samplers = [None]
    collate_fn =  misa_pretrain_collate_fn
    data_loader = create_loader(datasets, samplers, batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[collate_fn])[0]
    ######################################################################## Model ############################################################     
    logger.info("Creating Model")
    model = AcFormerPretrain(config=config)
    model = model.to(device)   
        
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    writer = SummaryWriter(log_dir='./output/tensorboard/')

    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']                       
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch']+1         
        else:
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)   
            m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)  
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped       
            state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped               
        model.load_state_dict(state_dict)    
        logger.info('load checkpoint from %s'%args.checkpoint)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
            
    ######################################################################## Training ###########################################################    
    logger.info("Start training")
    start_time = time.time()
    save_log = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+args.exp+'.log'
    for epoch in range(start_epoch, max_epoch):        
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)         
        train_stats = train(model, data_loader, optimizer, epoch, warmup_steps, device, lr_scheduler, writer, config, args)
        if False:
            save_feats = os.path.join(config['feat_save_dir'], str(epoch)+'_'+'features.pkl')
            eval(model, data_loader, optimizer, epoch, warmup_steps, device, lr_scheduler, save_feats, config, args)    
            logger.info(f"{save_feats} has saved.")
        if utils.is_main_process():  
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                     
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
                }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  
            
            with open(os.path.join(args.output_dir, save_log),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier()  
    writer.flush()           
    writer.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str)) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/test.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='./output/')

    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument("--local_rank", default=0, type=int)
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
    parser.add_argument('--data', type=str, default='pretrain_msa')

    # experiment
    parser.add_argument('--exp', type=str, default='pretrain') 

    # parser arguments
    args = parser.parse_args()
    print("Dataset {}".format(args.data))

    if args.data == "mosi":
        args.num_classes = 1  # 
    elif args.data == "mosei":
        args.num_classes = 1
    elif args.data == "ur_funny":
        args.num_classes = 2
    else:
        # print("No dataset mentioned")
        # exit()
        pass

    #  config definitations
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    

    
    main(args, config)