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

import torch
from torch import nn
import torch.nn.functional as F

class AcFormerPretrain(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer 
        self.distill = config['distill']
        self.output_size = output_size = config['num_classes']
        self.dropout_rate = dropout_rate = config['dropout']
        
        if config['loss_type']=='mse':
            self.criterion  = nn.MSELoss(reduction="mean")
        elif config['loss_type']=='bce':
            self.criterion  = nn.CrossEntropyLoss(reduction="mean")

        visual_encoder_layer = nn.TransformerEncoderLayer(d_model=config['embed_dim'], nhead=4)
        self.visual_encoder = nn.TransformerEncoder(visual_encoder_layer, num_layers=4)

        audio_encoder_layer = nn.TransformerEncoderLayer(d_model=config['embed_dim'], nhead=4)
        self.audio_encoder = nn.TransformerEncoder(audio_encoder_layer, num_layers=4)

        bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)

        encoder_layer = nn.TransformerEncoderLayer(d_model=config['multimodal_hidden_dim'], nhead=2)
        self.multimodal_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.vision_proj = nn.Linear(config['vision_width'], config['embed_dim'])
        self.audio_proj = nn.Linear(config['audio_width'], config['embed_dim'])
        
        # text_width = self.text_encoder.config.hidden_size
        # self.text_proj = nn.Linear(text_width, config['embed_dim'])
        self.fusion_head = nn.Sequential(
            nn.Linear(in_features=self.config['multimodal_hidden_dim'], out_features=self.config['embed_dim']),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(in_features=self.config['embed_dim'], out_features=output_size))

        if self.distill:
            self.visual_encoder_m = nn.TransformerEncoder(visual_encoder_layer, num_layers=1)
            self.audio_encoder_m = nn.TransformerEncoder(audio_encoder_layer, num_layers=1)                                       
            self.text_encoder_m = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)     
            self.multimodal_transformer_encoder_m = nn.TransformerEncoder(encoder_layer, num_layers=1)

            self.fusion_head_m = nn.Sequential(
                nn.Linear(in_features=self.config['multimodal_hidden_dim'], out_features=self.config['embed_dim']),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(in_features=self.config['embed_dim'], out_features=output_size))

            self.model_pairs = [
                                [self.visual_encoder,self.visual_encoder_m],
                                [self.audio_encoder,self.audio_encoder_m],
                                [self.text_encoder,self.text_encoder_m],
                                [self.fusion_head,self.fusion_head_m],
                                ]
            self.copy_params()
            self.momentum = 0.995
            
    def forward(self, sentences, visual, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask, targets, alpha=0, train=True):
        image_embed = self.vision_proj(visual)
        audio_embed = self.audio_proj(acoustic)
        self.image_embeds = self.visual_encoder(image_embed) # self.image_embeds torch.Size([40, 64, 768])  
        self.audio_embeds = self.audio_encoder(audio_embed)  # self.audio_embeds.shape torch.Size([40, 64, 768])
        
        bert_output = self.text_encoder(input_ids=bert_sent, attention_mask=bert_sent_mask, token_type_ids=bert_sent_type)
        bert_output = bert_output[0]
        bert_output = F.normalize(bert_output, dim=-1) 
        masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)        
        self.text_embeds = masked_output.permute(1,0,2) # self.text_embed.shape torch.Size([42,64,768])

        union_embeds = torch.cat((self.image_embeds, self.audio_embeds, self.text_embeds),dim=0) # torch.Size([40+40+40, 64, 768])
        encoder_output = self.multimodal_transformer_encoder(union_embeds)
        encoder_output_mean = torch.mean(encoder_output, dim=0)             
        
        if train:
            prediction = self.fusion_head(encoder_output_mean)
            if self.distill:                
                with torch.no_grad():
                    self._momentum_update()
                    # image_embeds_m = self.visual_encoder_m(image)
                    # output_m = self.text_encoder_m(text.input_ids,
                    #                            attention_mask = text.attention_mask,
                    #                            encoder_hidden_states = image_embeds_m,
                    #                            encoder_attention_mask = image_atts,
                    #                            return_dict = True
                    #                           )           
                    # prediction_m = self.cls_head_m(output_m.last_hidden_state[:,0,:])   
                    self.image_embeds_m = self.visual_encoder_m(image_embed)
                    self.audio_embeds_m = self.audio_encoder_m(audio_embed)
                    bert_output = self.text_encoder_m(input_ids=bert_sent, attention_mask=bert_sent_mask, token_type_ids=bert_sent_type)
                    bert_output = bert_output[0]
                    bert_output = F.normalize(bert_output, dim=-1) 
                    masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)        
                    self.text_embeds_m = masked_output.permute(1,0,2) # self.text_embed.shape torch.Size([42,64,768])
                    
                    union_embeds = torch.cat((self.image_embeds_m, self.audio_embeds_m, self.text_embeds_m),dim=0) # torch.Size([40+40+40, 64, 768])
                    encoder_output = self.multimodal_transformer_encoder_m(union_embeds)
                    encoder_output_mean = torch.mean(encoder_output, dim=0)                         
                    prediction_m = self.fusion_head_m(encoder_output_mean)
                loss = (1-alpha)*self.criterion(prediction, targets) + alpha*self.criterion(prediction, prediction_m)
            else:
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

