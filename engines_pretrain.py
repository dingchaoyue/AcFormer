import utils
import torch
import distributed
import pickle
from collections import defaultdict

def train(model, data_loader, optimizer, epoch, warmup_steps, device, scheduler, writer, config, args):
    # train
    model.train()  
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_vat', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_vat_m', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size
    
    if args.distributed:
        data_loader.sampler.set_epoch(epoch)
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        optimizer.zero_grad()
        video, video_aug, audio, audio_key_padding_mask, bert_sentences, bert_sentence_types, bert_sentence_att_mask, targets = batch 
        video = video.to(device,non_blocking=True) 
        video_aug = video_aug.to(device,non_blocking=True)
        audio = audio.to(device, non_blocking = True)
        audio_key_padding_mask = audio_key_padding_mask.to(device, non_blocking = True)
        bert_sentences, bert_sentence_types, bert_sentence_att_mask = bert_sentences.to(device), bert_sentence_types.to(device), bert_sentence_att_mask.to(device),             
        targets = targets.to(device, non_blocking =True)
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader)) 
        loss_vat, loss_vat_m, loss_i2a, loss_a2i, loss_i2t, loss_t2i, loss_a2t, loss_t2a, loss_i2i, loss_a2a, loss_t2t = model(video, video_aug, audio, audio_key_padding_mask, bert_sentences, bert_sentence_types, bert_sentence_att_mask, alpha = alpha, is_train=True)          
        loss = loss_vat + loss_vat_m

        writer.add_scalar("Loss/total_loss", loss, epoch)
        writer.add_scalars("Loss/cross_modal_alignment", {'ia':loss_i2a+loss_a2i, 'it':loss_i2t+loss_t2i, 'at':loss_a2t+loss_t2a}, epoch)
        writer.add_scalars("Loss/intra_modal_contrastive", {'ii':loss_i2i, 'aa': loss_a2a, 'tt':loss_t2t}, epoch)

        loss.backward()
        optimizer.step()
        
        metric_logger.update(loss_vat=loss_vat.item())
        metric_logger.update(loss_vat_m=loss_vat_m.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])         
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)                 
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def eval(model, data_loader, optimizer, epoch, warmup_steps, device, scheduler, save_feats, config, args):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    print_freq = 50
    image_feats = []
    audio_feats = []
    text_feats = []
    labels = []
    # step_size = 100
    header = 'Eval Epoch: [{}]'.format(epoch)        
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        video, video_aug, audio, audio_key_padding_mask, bert_sentences, bert_sentence_types, bert_sentence_att_mask, target = batch 
        video = video.to(device,non_blocking=True) 
        video_aug = video_aug.to(device,non_blocking=True)
        audio = audio.to(device, non_blocking = True)
        audio_key_padding_mask = audio_key_padding_mask.to(device, non_blocking = True)
        bert_sentences, bert_sentence_types, bert_sentence_att_mask = bert_sentences.to(device), bert_sentence_types.to(device), bert_sentence_att_mask.to(device),             
        target = target.to(device, non_blocking =True)
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))        
        image_feat, audio_feat, text_feat = model(video, video_aug, audio, audio_key_padding_mask, bert_sentences, bert_sentence_types, bert_sentence_att_mask, alpha = alpha, is_train=False)
        if args.distributed:
            # gathter data from multiple gpus
            image_feat_d, audio_feat_d, text_feat_d, target_d = distributed.all_gather([image_feat, audio_feat, text_feat, target])
            image_feats.append(image_feat_d.detach().cpu().numpy())
            audio_feats.append(audio_feat_d.detach().cpu().numpy())        
            text_feats.append(text_feat_d.detach().cpu().numpy())        
            labels.append(target_d.detach().cpu().numpy())
        else:
            print("Not implement Error!")
            exit()
    
    if utils.is_main_process():
        feature_dict = defaultdict(list)
        for class_id, visual_feat, acoustic_feat, linguistic_feat in zip(labels, image_feats, audio_feats, text_feats):
            for cls_id, vis_feat, audio_feat, txt_feat in zip (class_id, visual_feat, acoustic_feat, linguistic_feat):
                feature_dict[int(cls_id)].append([vis_feat, audio_feat, txt_feat])
        with open(save_feats, 'wb') as f:
            pickle.dump(feature_dict, f)

    metric_logger.synchronize_between_processes()
    






