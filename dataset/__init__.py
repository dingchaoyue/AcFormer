import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.caption_dataset import re_train_dataset, re_eval_dataset, pretrain_dataset
from dataset.nlvr_dataset import nlvr_dataset
from dataset.ve_dataset import ve_dataset
from dataset.vqa_dataset import vqa_dataset
from dataset.msa_dataset import MSADataset, MSAPretrainDataset
from dataset.msa_raw_dataset import RawMosiDataset

from dataset.grounding_dataset import grounding_dataset

from dataset.randaugment import RandomAugment
from dataset.utils import GaussianBlur

from transformers import *
from dataset.create_dataset import PAD
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def get_key_padding_mask(padded_input, pad_idx):
    """Creates a binary mask to prevent attention to padded locations.
    Arguments
    ----------
    padded_input: int
        Padded input.
    pad_idx:
        idx for padding element.
    Example
    -------
    >>> a = torch.LongTensor([[1,1,0], [2,3,0], [4,5,0]])
    >>> get_key_padding_mask(a, pad_idx=0)
    tensor([[False, False,  True],
            [False, False,  True],
            [False, False,  True]])
    """
    if len(padded_input.shape) == 4:
        bz, time, ch1, ch2 = padded_input.shape
        padded_input = padded_input.reshape(bz, time, ch1 * ch2)

    key_padded_mask = padded_input.eq(pad_idx).to(padded_input.device)

    # if the input is more than 2d, mask the locations where they are silence
    # across all channels
    if len(padded_input.shape) > 2:
        key_padded_mask = key_padded_mask.float().prod(dim=-1).bool()
        return key_padded_mask.detach()

    return key_padded_mask.detach()


def length_to_mask(length, max_len=None, dtype=None, device=None):
    """Creates a binary mask for each sequence.
    Reference: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3
    Arguments
    ---------
    length : torch.LongTensor
        Containing the length of each sequence in the batch. Must be 1D.
    max_len : int
        Max length for the mask, also the size of the second dimension.
    dtype : torch.dtype, default: None
        The dtype of the generated mask.
    device: torch.device, default: None
        The device to put the mask variable.
    Returns
    -------
    mask : tensor
        The binary mask.
    Example
    -------
    >>> length=torch.Tensor([1,2,3])
    >>> mask=length_to_mask(length)
    >>> mask
    tensor([[1., 0., 0.],
            [1., 1., 0.],
            [1., 1., 1.]])
    """
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long().item()  # using arange to generate mask
    mask = torch.arange(
        max_len, device=length.device, dtype=length.dtype
    ).expand(len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    if device is None:
        device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask

def get_lookahead_mask(padded_input):
    """Creates a binary mask for each sequence which maskes future frames.
    Arguments
    ---------
    padded_input: torch.Tensor
        Padded input tensor.
    Example
    -------
    >>> a = torch.LongTensor([[1,1,0], [2,3,0], [4,5,0]])
    >>> get_lookahead_mask(a)
    tensor([[0., -inf, -inf],
            [0., 0., -inf],
            [0., 0., 0.]])
    """
    seq_len = padded_input.shape[1]
    mask = (
        torch.triu(torch.ones((seq_len, seq_len), device=padded_input.device))
        == 1
    ).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask.detach().to(padded_input.device)


def make_masks_transformer( src, tgt, wav_len=None, pad_idx=0):
    """This method generates the masks for training the transformer model.
    Arguments
    ---------
    src : tensor
        The sequence to the encoder (required).
    tgt : tensor
        The sequence to the decoder (required).
    pad_idx : int
        The index for <pad> token (default=0).
    """
    src_key_padding_mask = None
    if wav_len is not None:
        abs_len = torch.round(wav_len * src.shape[1])
        src_key_padding_mask = ~length_to_mask(abs_len).bool()

    tgt_key_padding_mask = get_key_padding_mask(tgt, pad_idx=pad_idx)

    src_mask = None
    tgt_mask = get_lookahead_mask(tgt)
    return src_key_padding_mask, tgt_key_padding_mask, src_mask, tgt_mask

 
def make_masks(src, wav_len=None, pad_idx=0):
    """This method generates the masks for training the transformer model.
    Arguments
    ---------
    src : tensor
        The sequence to the encoder (required).
    pad_idx : int
        The index for <pad> token (default=0).
    """
    src_key_padding_mask = None
    if wav_len is not None:
        abs_len = torch.round(wav_len * src.shape[1])
        src_key_padding_mask = ~length_to_mask(abs_len).bool()
    return src_key_padding_mask


def msa_raw_collate_fn(batch):
    '''
    Collate functions assume batch = [Dataset[i] for i in index_set]
    '''
    # for later use we sort the batch in descending order of length
    batch = sorted(batch, key=lambda x: len(x['text']), reverse=True)
    # is an alias for the default tensor type torch.FloatTensor(), 
    # used to generate a tensor of single-precision floating-point type.
    targets = torch.Tensor([sample['label'] for sample in batch]).unsqueeze(1)
    video_data = torch.stack([torch.FloatTensor(sample['video']) for sample in batch])  
    audio_data = torch.stack([torch.FloatTensor(sample['audio']) for sample in batch])
    wav_lens = torch.Tensor([sample['wav_lens'] for sample in batch])
    audio_key_padding_mask = ~length_to_mask(wav_lens, max_len=1024).bool() # mask send to audio transformer

    # create bert indices using tokenizer
    bert_details = []
    SENT_LEN = len(batch[0]['text'])
    for sample in batch:
        text = " ".join(sample['text'])
        encoded_bert_sent = tokenizer.encode_plus(
                        text,
                        add_special_tokens = False,   
                        max_length = SENT_LEN,    
                        padding = 'max_length',       
                        truncation = True
                   )
        bert_details.append(encoded_bert_sent)

    # Bert things are batch_first
    bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
    bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
    bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])


    return video_data, audio_data, audio_key_padding_mask, bert_sentences, bert_sentence_types, bert_sentence_att_mask, targets


def misa_pretrain_collate_fn(batch):
    '''
    Collate functions assume batch = [Dataset[i] for i in index_set]
    '''
    # for later use we sort the batch in descending order of length
    batch = sorted(batch, key=lambda x: len(x['text']), reverse=True)
    # is an alias for the default tensor type torch.FloatTensor(), 
    # used to generate a tensor of single-precision floating-point type.
    targets = torch.Tensor([sample['label'] for sample in batch]).unsqueeze(1)
    video_data = torch.stack([torch.FloatTensor(sample['video']) for sample in batch])
    video_data_aug =   torch.stack([torch.FloatTensor(sample['video_aug']) for sample in batch])
    audio_data = pad_sequence([torch.FloatTensor(sample['audio']) for sample in batch], batch_first=True)
    wav_lens = torch.IntTensor([sample['wav_lens'] for sample in batch])
    max_len = torch.max(wav_lens).item()
    audio_key_padding_mask = ~length_to_mask(wav_lens, max_len=max_len).bool() # mask send to audio transformer
    
    ## BERT-based features input prep

    SENT_LEN = len(batch[0]['text'])
    # create bert indices using tokenizer
    bert_details = []
    for sample in batch:
        text = " ".join(sample['text'])
        encoded_bert_sent = tokenizer.encode_plus(
                        text,
                        add_special_tokens = True,   
                        max_length = SENT_LEN + 2,    
                        padding = 'max_length',       
                        truncation = True)
        bert_details.append(encoded_bert_sent)

    # Bert things are batch_first
    bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
    bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
    bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])

    return video_data, video_data_aug, audio_data, audio_key_padding_mask, bert_sentences, bert_sentence_types, bert_sentence_att_mask, targets


def misa_collate_fn(batch):
    '''
    Collate functions assume batch = [Dataset[i] for i in index_set]
    '''
    # for later use we sort the batch in descending order of length
    batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)  # 16 *
    
    # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
    labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)  # 
    sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)  # torch.Size([52, 32])
    visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])  # ([52, 32, 75])
    acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])  # ([52, 32, 81])
    visual_lens = torch.IntTensor([sample[0][1].shape[0] for sample in batch])
    acoustic_lens = torch.IntTensor([sample[0][2].shape[0] for sample in batch])
    max_visual_len = torch.max(visual_lens).item()
    max_acoustic_len = torch.max(acoustic_lens).item()
    visual_key_padding_mask = ~length_to_mask(visual_lens, max_len=max_visual_len).bool() # mask send to audio transformer
    audio_key_padding_mask = ~length_to_mask(acoustic_lens, max_len=max_acoustic_len).bool() # mask send to audio transformer

    ## BERT-based features input prep
    SENT_LEN = sentences.size(0)
    # create bert indices using tokenizer
    bert_details = []
    for sample in batch:
        text = " ".join(sample[0][3])
        encoded_bert_sent = tokenizer.encode_plus(
                        text,                          # Sentence to encode.
                        add_special_tokens = True,    # Add '[CLS]' and '[SEP]'
                        max_length = SENT_LEN+2,         # Pad & truncate all sentences.
                        padding = 'max_length',       # 补全操作
                        truncation = True,            # 截断操作
                        # return_attention_mask = True, # Construct attn. masks.
                        # return_tensors = 'pt',        # Return pytorch tensors.
                    )
        bert_details.append(encoded_bert_sent)

    # Bert things are batch_first
    bert_sentence_ids = torch.LongTensor([sample["input_ids"] for sample in bert_details])
    bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
    bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])
    
    # lengths are useful later in using RNNs
    lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])

    return sentences, visual, visual_key_padding_mask, acoustic, audio_key_padding_mask, labels, lengths, bert_sentence_ids, bert_sentence_types, bert_sentence_att_mask


def create_dataset(dataset, config):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    # jinyu: add augmentation
    pretrain_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.2, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])       
    # jinyu: add augmentation
    train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])  
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])   
    
    if dataset=='pretrain':        
        dataset = pretrain_dataset(config['train_file'], pretrain_transform)                  
        return dataset      
               
    elif dataset=='re':     
        train_dataset = re_train_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = re_eval_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset   

    elif dataset=='vqa': 
        train_dataset = vqa_dataset(config['train_file'], train_transform, config['vqa_root'], config['vg_root'], split='train') 
        vqa_test_dataset = vqa_dataset(config['test_file'], test_transform, config['vqa_root'], config['vg_root'], split='test', answer_list=config['answer_list'])       
        return train_dataset, vqa_test_dataset

    elif dataset=='nlvr':   
        train_dataset = nlvr_dataset(config['train_file'], train_transform, config['image_root'])  
        val_dataset = nlvr_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = nlvr_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset        
               
    elif dataset=='ve':   
        train_dataset = ve_dataset(config['train_file'], train_transform, config['image_root'])  
        val_dataset = ve_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = ve_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset     
    
    elif dataset=='grounding':
        train_transform = transforms.Compose([                        
                transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])         
        train_dataset = grounding_dataset(config['train_file'], train_transform, config['image_root'], mode='train')       
        test_dataset = grounding_dataset(config['test_file'], test_transform, config['image_root'], mode='test')             
        return train_dataset, test_dataset
        
    elif (dataset == "mosi" or dataset == 'mosei' or dataset == 'ur_funny'):        
        train_dataset= MSADataset(config,split='train')
        val_dataset = MSADataset(config,split='dev')
        test_dataset = MSADataset(config,split='test')
        return train_dataset, val_dataset, test_dataset
    
    elif dataset == "pretrain_msa":
        pretrain_transform = transforms.Compose([                        
                transforms.RandomResizedCrop(config['image_res'],scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                # RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                #                                 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])   
        dataset = MSAPretrainDataset(config, pretrain_transform=pretrain_transform, split='train')
        return dataset
    
    elif dataset == 'mosi_raw':
        train_dataset= RawMosiDataset(config,split='train')
        val_dataset = RawMosiDataset(config,split='dev')
        test_dataset = RawMosiDataset(config,split='test')
        return train_dataset, val_dataset, test_dataset

def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n

def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = True

        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders