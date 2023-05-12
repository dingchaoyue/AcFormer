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
from models.attention_bottleneck_transformer import TransformerEncoderLayer, AttentionBottleneckTransformer

import torch
from torch import nn
import torch.nn.functional as F

class MBT(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer 
        self.distill = config['distill']
        output_size = config['num_classes']
        dropout_rate = config['dropout']
        embed_dim = config['embed_dim']
        neck_size = config['neck_size']
        num_layers = config['num_layers']
        fusion_layer = config['fusion_layer']

        if config['loss_type']=='mse':
            self.criterion  = nn.MSELoss(reduction="mean")
        elif config['loss_type']=='bce':
            self.criterion  = nn.CrossEntropyLoss(reduction="mean")
        
        UNI_visual_encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=4)
        UNI_audio_encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=4)
        UNI_text_encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=4)
        MUL_visual_encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=4)
        MUL_audio_encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=4)
        MUL_text_encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=4)
        self.multimodal_transformer_encoder = AttentionBottleneckTransformer(
            UNI_visual_encoder_layer, UNI_audio_encoder_layer, UNI_text_encoder_layer,
            MUL_visual_encoder_layer, MUL_audio_encoder_layer, MUL_text_encoder_layer,
            num_layers, fusion_layer, neck_size, embed_dim)

        bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.pre_text_encoder = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
        self.vision_proj = nn.Linear(config['vision_width'], config['embed_dim'])
        self.audio_proj = nn.Linear(config['audio_width'], config['embed_dim'])

        self.fusion_head = nn.Sequential(
            nn.Linear(in_features=self.config['multimodal_hidden_dim'], out_features=self.config['embed_dim']),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(in_features=self.config['embed_dim'], out_features=output_size))
            
    def forward(self, sentences, visual, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask, targets, alpha=0, train=True):
        self.image_embed = self.vision_proj(visual) # self.image_embeds torch.Size([40, 64, 768])
        self.audio_embed = self.audio_proj(acoustic) # self.audio_embeds.shape torch.Size([40, 64, 768])  
        with torch.no_grad():    
            bert_output = self.pre_text_encoder(input_ids=bert_sent, attention_mask=bert_sent_mask, token_type_ids=bert_sent_type)
            bert_output = bert_output[0]
            bert_output = F.normalize(bert_output, dim=-1) 
            masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)        
            self.text_embeds = masked_output.permute(1,0,2) # self.text_embed.shape torch.Size([40,64,768])
        
        output_v, output_a, output_t, shared_neck = self.multimodal_transformer_encoder(self.image_embed, self.audio_embed, self.text_embeds)
        union_embeds = torch.cat((output_v, output_a, output_t, shared_neck),dim=0) # torch.Size([40+40+40+12, 64, 768])
        encoder_output_mean = torch.mean(union_embeds, dim=0)
    
        if train:
            prediction = self.fusion_head(encoder_output_mean)
            loss = self.criterion(prediction, targets)
            return loss
        else:
            prediction = self.fusion_head(encoder_output_mean)
            return prediction

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

