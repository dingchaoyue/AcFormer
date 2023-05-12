import torch
import utils
import numpy as np
from tqdm import tqdm
from tqdm import tqdm_notebook
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from scipy.special import expit
from distributed import is_primary, all_gather
import distributed

def raw_train(model, data_loader, optimizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
 
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        vision, audio, audio_key_padding_mask, bert_sent, bert_sent_type, bert_sent_mask, target = batch
        vision, audio, audio_key_padding_mask, target = vision.to(device), audio.to(device), audio_key_padding_mask.to(device), target.to(device) 
        bert_sent, bert_sent_type, bert_sent_mask = bert_sent.to(device), bert_sent_type.to(device), bert_sent_mask.to(device)
        
        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))
        loss = model(vision, audio, audio_key_padding_mask, bert_sent, bert_sent_type, bert_sent_mask, targets=target, train=True)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
               
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    


@torch.no_grad()
def raw_evaluate(model, data_loader, device, args, to_print=True):
    # eval
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    y_true, y_pred = [], []
    header = 'Evaluation:'
    print_freq = 50
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        vision, audio, audio_key_padding_mask, bert_sent, bert_sent_type, bert_sent_mask, target = batch
        vision, audio, audio_key_padding_mask, target = vision.to(device), audio.to(device), audio_key_padding_mask.to(device), target.to(device) 
        bert_sent, bert_sent_type, bert_sent_mask = bert_sent.to(device), bert_sent_type.to(device), bert_sent_mask.to(device)  
        ####################################################### Inference #####################################################
        prediction = model(vision, audio, audio_key_padding_mask, bert_sent, bert_sent_type, bert_sent_mask, targets=target, train=False)
        # if args.distributed and is_primary(args):
        # if args.distributed and utils.is_main_process():
        if args.distributed:
            # gathter data from multiple gpus
            pred, label = distributed.all_gather([prediction, target])
            y_pred.append(pred.detach().cpu().numpy())
            y_true.append(label.detach().cpu().numpy())
        else:        
            y_pred.append(prediction.detach().cpu().numpy())
            y_true.append(target.detach().cpu().numpy())
            
    # measurement
    y_true = np.concatenate(y_true, axis=0).squeeze()
    y_pred = np.concatenate(y_pred, axis=0).squeeze()
    acc = calc_metrics(y_true, y_pred, to_print)
    # metric_logger.meters['acc'].update(accuracy.item(), n=image0.size(0))
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {'acc':acc}



def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, args):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
 
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        text, vision, visual_key_padding_mask, audio, audio_key_padding_mask, target, length, bert_sent, bert_sent_type, bert_sent_mask = batch
        # images = torch.cat([image0, image1], dim=0)
        # images, targets = images.to(device), targets.to(device)   
        text, vision, audio, target, length = text.to(device), vision.to(device), audio.to(device), target.to(device), length.to(device)
        bert_sent, bert_sent_type, bert_sent_mask = bert_sent.to(device), bert_sent_type.to(device), bert_sent_mask.to(device)
        audio_key_padding_mask = audio_key_padding_mask.to(device)
        visual_key_padding_mask = visual_key_padding_mask.to(device)
        if args.data == 'ur_funny':
            target = target.squeeze()                
        # text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)
        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))
            # loss = model(images, text_inputs, targets=targets, train=True, alpha=alpha)    
        loss = model(text, vision, visual_key_padding_mask, audio, audio_key_padding_mask, length, bert_sent, bert_sent_type, bert_sent_mask, targets=target, alpha=alpha, train=True)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
               
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    

@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device, config, args, to_print=True):
    # eval
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    y_true, y_pred = [], []
    header = 'Evaluation:'
    print_freq = 50
    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, print_freq, header):
            text, vision, visual_key_padding_mask, audio, audio_key_padding_mask, target, length, bert_sent, bert_sent_type, bert_sent_mask = batch
            text, vision, audio, target, length = text.to(device), vision.to(device), audio.to(device), target.to(device), length.to(device)
            bert_sent, bert_sent_type, bert_sent_mask = bert_sent.to(device), bert_sent_type.to(device), bert_sent_mask.to(device)
            if args.data == 'ur_funny':
                target = target.squeeze()             
            ###################################################### Inference #####################################################
            prediction = model(text, vision, visual_key_padding_mask, audio, audio_key_padding_mask, length, bert_sent, bert_sent_type, bert_sent_mask, targets=target, train=False)
            # if args.distributed and is_primary(args):
            # if args.distributed and utils.is_main_process():
            if args.distributed:
                # gathter data from multiple gpus
                pred, label = distributed.all_gather([prediction, target])
                y_pred.append(pred.detach().cpu().numpy())
                y_true.append(label.detach().cpu().numpy())
            else:        
                y_pred.append(prediction.detach().cpu().numpy())
                y_true.append(target.detach().cpu().numpy())
    # measurement
    y_true = np.concatenate(y_true, axis=0).squeeze()
    y_pred = np.concatenate(y_pred, axis=0).squeeze()
    acc = calc_metrics(y_true, y_pred, args, to_print)
    # metric_logger.meters['acc'].update(accuracy.item(), n=image0.size(0))
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {'acc':acc}


def calc_metrics(y_true, y_pred, args, to_print=False):
    """
    Metric scheme adapted from:
    https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
    """
    if args.data == "ur_funny":
        test_preds = np.argmax(y_pred, 1)
        test_truth = y_true
        if to_print:
            print("Confusion Matrix (pos/neg) :")
            print(confusion_matrix(test_truth, test_preds))
            print("Classification Report (pos/neg) :")
            print(classification_report(test_truth, test_preds, digits=5))
            print("Accuracy (pos/neg) ", accuracy_score(test_truth, test_preds))        
        return accuracy_score(test_truth, test_preds)
    else:
        test_truth = y_true
        test_preds = y_pred
        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        
        test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
        test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
        test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

        mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
        mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
        f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
        
        # pos - neg
        binary_truth = (test_truth[non_zeros] > 0)
        binary_preds = (test_preds[non_zeros] > 0)

        if to_print:
            print("mae: ", mae)
            print("corr: ", corr)
            print("mult_acc: ", mult_a7)
            print("Classification Report (pos/neg) :")
            print(classification_report(binary_truth, binary_preds, digits=5))
            print("Accuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))
        
        # non-neg - neg
        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)

        if to_print:
            print("Classification Report (non-neg/neg) :")
            print(classification_report(binary_truth, binary_preds, digits=5))
            print("Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))    
        return accuracy_score(binary_truth, binary_preds)


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth
    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label==1) & (predicted_label==1)))
    tn = float(np.sum((true_label==0) & (predicted_label==0)))
    p = float(np.sum(true_label==1))
    n = float(np.sum(true_label==0))

    return (tp * (n/p) +tn) / (2*n)


def eval_iemocap(results, truths, single=-1):
    emos = ["Neutral", "Happy", "Sad", "Angry"]
    if single < 0:
        test_preds = results.view(-1, 4, 2).cpu().detach().numpy()
        test_truth = truths.view(-1, 4).cpu().detach().numpy()
        
        for emo_ind in range(4):
            print(f"{emos[emo_ind]}: ")
            test_preds_i = np.argmax(test_preds[:,emo_ind],axis=1)
            test_truth_i = test_truth[:,emo_ind]
            f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
            acc = accuracy_score(test_truth_i, test_preds_i)
            print("  - F1 Score: ", f1)
            print("  - Accuracy: ", acc)
    else:
        test_preds = results.view(-1, 2).cpu().detach().numpy()
        test_truth = truths.view(-1).cpu().detach().numpy()
        
        print(f"{emos[single]}: ")
        test_preds_i = np.argmax(test_preds,axis=1)
        test_truth_i = test_truth
        f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
        acc = accuracy_score(test_truth_i, test_preds_i)
        print("  - F1 Score: ", f1)
        print("  - Accuracy: ", acc)