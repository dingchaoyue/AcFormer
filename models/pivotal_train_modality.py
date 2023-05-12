import random
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from functools import partial
# from models.ast import AudioTransformer
# from models.vit import VisionTransformer
from transformers import BertModel, BertConfig
from torch.nn.modules.normalization import LayerNorm
from models.attention_pivotal_transformer import  AttentionPivotalTransformer, \
        TransformerEncoderLayer, TransformerDecoderLayer
from models.attention_pivotal_raw_transformer import RawPivotalTransformer

import torch
from torch import nn
import torch.nn.functional as F

class RawModalPA(nn.Module):
    def __init__(self, config = None):
        super().__init__()
        self.config = config
        self.distill = config['distill']
        output_size = config['num_classes']
        dropout_rate = config['dropout']
        embed_dim = config['embed_dim']        
        neck_size = config['neck_size']
        num_layers = config['num_layers']
        start_fusion_layer = config['start_fusion_layer']
        fusion_head = config['fusion_head']
        img_size = config['image_size']
        num_frames = config['num_frames']
        positional_embedding = config['positional_embedding']
        freq_bins = config['freq_bins']
        time_frames = config['time_frames']
        self.modality = modality = config['modality']

        if config['loss_type']=='mse':
            self.criterion  = nn.MSELoss(reduction="mean")
        elif config['loss_type']=='bce':
            self.criterion  = nn.CrossEntropyLoss(reduction="mean")
    
        self.RPT = RawPivotalTransformer(img_size, num_frames, freq_bins, time_frames,
                                                  positional_embedding, num_layers, start_fusion_layer, 
                                                  neck_size, embed_dim, fusion_head, modality)

        bertconfig = BertConfig.from_pretrained('bert-base-uncased')
        self.pre_text_encoder = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)                     
        
        if self.modality in ['video','video+audio', 'video+text']:
            self.pred_head_v = nn.Sequential(
                # nn.Linear(in_features=self.config['embed_dim'], out_features=self.config['embed_dim']),
                # nn.Dropout(p=dropout_rate),
                # nn.ReLU(),
                nn.Linear(in_features=self.config['embed_dim'], out_features=output_size))
        
        if self.modality in ['audio', 'video+audio', 'audio+text']:
            self.pred_head_a = nn.Sequential(
                # nn.Linear(in_features=self.config['embed_dim'], out_features=self.config['embed_dim']),
                # nn.Dropout(p=dropout_rate),
                # nn.ReLU(),
                nn.Linear(in_features=self.config['embed_dim'], out_features=output_size))
        
        if self.modality in ['text','video+text', 'audio+text']:            
            self.pred_head_t = nn.Sequential(
                # nn.Linear(in_features=self.config['embed_dim'], out_features=self.config['embed_dim']),
                # nn.Dropout(p=dropout_rate),
                # nn.ReLU(),
                nn.Linear(in_features=self.config['embed_dim'], out_features=output_size))
        
        if self.modality in ['video+audio', 'video+text', 'audio+text']:
            self.pred_head_union = nn.Sequential(
                # nn.Linear(in_features=self.config['embed_dim'], out_features=self.config['embed_dim']),
                # nn.Dropout(p=dropout_rate),
                # nn.ReLU(),
                nn.Linear(in_features=self.config['embed_dim'], out_features=output_size))

    def forward(self, video_data, audio_data, audio_key_padding_mask, bert_sent, bert_sent_type, bert_sent_mask, targets, train=True):        
        with torch.no_grad():    
            bert_output = self.pre_text_encoder(input_ids=bert_sent, attention_mask=bert_sent_mask, token_type_ids=bert_sent_type)
            bert_output = bert_output[0]
            bert_output = F.normalize(bert_output, dim=-1) 
            masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)        
            text_embeds = masked_output.permute(1,0,2) # self.text_embed.shape torch.Size([40,64,768])
        
        # we can also consider the output sequnences [batch, max_len, embedding_dim], the average across max_len dimensions to get averaged/mean embeddings
        # Step 1: Expand Attention/Padding Mask from [batch_size, max_len] to [batch_size, max_len, hidden_size].
        # Step 2: Sum Embeddings along max_len axis so now we have [batch_size, hidden_size].
        # Step 3: Sum Mask along max_len axis. This is done so that we can ignore padding tokens.
        # Step 4: Take Average.
        if self.modality == 'text':
            output_t = self.RPT(video_data, audio_data, audio_key_padding_mask, text_embeds, bert_sent_mask)
            text_mask_expanded = bert_sent_mask.unsqueeze(-1).expand(output_t.size()).float()
            text_sum_embeddings = torch.sum(output_t * text_mask_expanded, axis=1)
            text_sum_mask = text_mask_expanded.sum(axis=1)
            text_sum_mask = torch.clamp(text_sum_mask, min=1e-9)
            text_mean_embeddings = text_sum_embeddings / text_sum_mask # torch.Size([1, 768]
            if train:
                pred_t = self.pred_head_t(text_mean_embeddings)
                loss_t = self.criterion(pred_t, targets)
                return loss_t
            else:
                pred_t = self.pred_head_t(text_mean_embeddings)
                return pred_t

        elif self.modality == 'audio':
            output_a = self.RPT(video_data, audio_data, audio_key_padding_mask, text_embeds, bert_sent_mask)
            audio_mask_expanded = (~audio_key_padding_mask).float().unsqueeze(-1).expand(output_a.size())
            audio_sum_embeddings = torch.sum(output_a * audio_mask_expanded, axis=1)
            audio_sum_mask = audio_mask_expanded.sum(axis=1)
            audio_sum_mask = torch.clamp(audio_sum_mask, min=1e-9)
            audio_mean_embeddings = audio_sum_embeddings / audio_sum_mask # torch.Size([1, 768]            
            if train:
                pred_a = self.pred_head_a(audio_mean_embeddings)                
                loss_a = self.criterion(pred_a, targets)
                return loss_a
            else:
                pred_a = self.pred_head_a(audio_mean_embeddings)
                return pred_a

        elif self.modality  == 'video':
            output_v = self.RPT(video_data, audio_data, audio_key_padding_mask, text_embeds, bert_sent_mask)            
            vision_mean_embeddings = torch.mean(output_v, axis=1) # torch.Size([1, 768]
            if train:
                pred_v = self.pred_head_v(vision_mean_embeddings)
                loss_v = self.criterion(pred_v, targets)
                return loss_v
            else:
                pred_v = self.pred_head_v(vision_mean_embeddings)
                return pred_v
            
        else:
            raise NotImplementedError


    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

