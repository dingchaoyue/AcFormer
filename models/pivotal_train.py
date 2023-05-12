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
from models.attention_pivotal_raw_transformer import RawAttentionPivotalTransformer

import torch
from torch import nn
import torch.nn.functional as F

class RawMPA(nn.Module):
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

        if config['loss_type']=='mse':
            self.criterion  = nn.MSELoss(reduction="mean")
        elif config['loss_type']=='bce':
            self.criterion  = nn.CrossEntropyLoss(reduction="mean")
    
        self.APT = RawAttentionPivotalTransformer(img_size, num_frames, freq_bins, time_frames,
                                                  positional_embedding, num_layers, start_fusion_layer, 
                                                  neck_size, embed_dim, fusion_head)

        bertconfig = BertConfig.from_pretrained('bert-base-uncased')
        self.pre_text_encoder = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)

        self.pred_head_v = nn.Sequential(
            # nn.Linear(in_features=self.config['embed_dim'], out_features=self.config['embed_dim']),
            # nn.Dropout(p=dropout_rate),
            # nn.ReLU(),
            nn.Linear(in_features=self.config['embed_dim'], out_features=output_size))
        self.pred_head_a = nn.Sequential(
            # nn.Linear(in_features=self.config['embed_dim'], out_features=self.config['embed_dim']),
            # nn.Dropout(p=dropout_rate),
            # nn.ReLU(),
            nn.Linear(in_features=self.config['embed_dim'], out_features=output_size))
        self.pred_head_t = nn.Sequential(
            # nn.Linear(in_features=self.config['embed_dim'], out_features=self.config['embed_dim']),
            # nn.Dropout(p=dropout_rate),
            # nn.ReLU(),
            nn.Linear(in_features=self.config['embed_dim'], out_features=output_size))
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
        # >>> old_version
        # union_embeds = torch.cat((output_v, output_a, output_t, shared_neck), dim = 0) # torch.Size([12544+1212+10+12, 64, 768])
        # encoder_output_mean = torch.mean(union_embeds, dim=0)
        # >>> new version
        # We can also consider the output sequnences [batch, max_len, embedding_dim], the average across max_len dimensions to get averaged/mean embeddings
        # Step 1: Expand Attention/Padding Mask from [batch_size, max_len] to [batch_size, max_len, hidden_size].
        # Step 2: Sum Embeddings along max_len axis so now we have [batch_size, hidden_size].
        # Step 3: Sum Mask along max_len axis. This is done so that we can ignore padding tokens.
        # Step 4: Take Average.
        output_v, output_a, output_t, shared_neck = self.APT(video_data, audio_data, audio_key_padding_mask, text_embeds, bert_sent_mask)
        text_mask_expanded = bert_sent_mask.unsqueeze(-1).expand(output_t.size()).float()
        audio_mask_expanded = (~audio_key_padding_mask).float().unsqueeze(-1).expand(output_a.size())
        text_sum_embeddings = torch.sum(output_t * text_mask_expanded, axis=1)
        audio_sum_embeddings = torch.sum(output_a * audio_mask_expanded, axis=1)
        text_sum_mask = text_mask_expanded.sum(axis=1)
        text_sum_mask = torch.clamp(text_sum_mask, min=1e-9)
        audio_sum_mask = audio_mask_expanded.sum(axis=1)
        audio_sum_mask = torch.clamp(audio_sum_mask, min=1e-9)
        text_mean_embeddings = text_sum_embeddings / text_sum_mask # torch.Size([1, 768]
        audio_mean_embeddings = audio_sum_embeddings / audio_sum_mask # torch.Size([1, 768]
        vision_mean_embeddings = torch.mean(output_v, axis=1) # torch.Size([1, 768]
        pivotal_mean_embeddings = torch.mean(shared_neck, axis=1) # torch.Size([1, 768]

        if train:
            pred_v = self.pred_head_v(vision_mean_embeddings)
            pred_a = self.pred_head_a(audio_mean_embeddings)
            pred_t = self.pred_head_t(text_mean_embeddings)
            pred_pivotal = self.pred_head_union(pivotal_mean_embeddings)
            loss_v = self.criterion(pred_v, targets)
            loss_a = self.criterion(pred_a, targets)
            loss_t = self.criterion(pred_t, targets)
            loss_pivotal = self.criterion(pred_pivotal, targets)
            # todo 
            # prediction = (pred_v + pred_a + pred_t + pred_pivotal) /4.0  
            # loss = self.criterion(prediction, targets)
            loss = loss_v + loss_a + loss_t + loss_pivotal
            return loss
        else:
            pred_v = self.pred_head_v(vision_mean_embeddings)
            pred_a = self.pred_head_a(audio_mean_embeddings)
            pred_t = self.pred_head_t(text_mean_embeddings)
            pred_pivotal = self.pred_head_union(pivotal_mean_embeddings)
            prediction = (pred_v + pred_a + pred_t + pred_pivotal) / 4.0 # equal contribution
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

class MPA(nn.Module):
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
        start_fusion_layer = config['start_fusion_layer']
        self.fusion_strategy = config['fusion_strategy']

        if config['loss_type']=='mse':
            self.criterion  = nn.MSELoss(reduction="mean")
            self.criterion2 = nn.L1Loss(reduction="mean")
        elif config['loss_type']=='bce':
            self.criterion  = nn.CrossEntropyLoss(reduction="mean")

        
        UNI_visual_encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=6, dim_feedforward=1536)
        UNI_audio_encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=6, dim_feedforward=1536)
        UNI_text_encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=6, dim_feedforward=1536)

        TOKEN_collection_vision_layer = TransformerDecoderLayer(d_model=embed_dim, nhead=6, dim_feedforward=1536)
        TOKEN_collection_audio_layer = TransformerDecoderLayer(d_model=embed_dim, nhead=6, dim_feedforward=1536)
        TOKEN_collection_text_layer = TransformerDecoderLayer(d_model=embed_dim, nhead=6, dim_feedforward=1536)
        TOKEN_propagation_vision_layer = TransformerDecoderLayer(embed_dim, nhead=6, dim_feedforward=1536)
        TOKEN_propagation_audio_layer = TransformerDecoderLayer(embed_dim, nhead=6, dim_feedforward=1536)
        TOKEN_propagation_text_layer = TransformerDecoderLayer(embed_dim, nhead=6, dim_feedforward=1536)

        self.multimodal_transformer_encoder = AttentionPivotalTransformer(
            UNI_visual_encoder_layer, UNI_audio_encoder_layer, UNI_text_encoder_layer,
            TOKEN_collection_vision_layer, TOKEN_collection_audio_layer, TOKEN_collection_text_layer,
            TOKEN_propagation_vision_layer, TOKEN_propagation_audio_layer, TOKEN_propagation_text_layer,
            num_layers, start_fusion_layer, neck_size, embed_dim)

        # bertconfig = BertConfig.from_pretrained('bert-base-uncased') # config=bertconfig
        self.pre_text_encoder = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)

        # old version        
        self.vision_proj = nn.Linear(config['vision_width'], config['embed_dim'])
        self.audio_proj = nn.Linear(config['audio_width'], config['embed_dim'])
        
        # new version
        # 1. Temporal Convolutional layers
        # self.proj_v = nn.Conv1d(config['vision_width'], config['embed_dim'], kernel_size=1, padding=0, bias=False)
        # self.proj_a = nn.Conv1d(config['audio_width'], config['embed_dim'], kernel_size=1, padding=0, bias=False)
        # self.proj_t = nn.Conv1d(config['text_width'], config['embed_dim'], kernel_size=1, padding=0, bias=False)
        
        if self.fusion_strategy == 'separate':
            self.pred_head_v = nn.Sequential(
                nn.Linear(in_features=self.config['embed_dim'], out_features=self.config['embed_dim']),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(in_features=self.config['embed_dim'], out_features=output_size))
            self.pred_head_a = nn.Sequential(
                nn.Linear(in_features=self.config['embed_dim'], out_features=self.config['embed_dim']),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(in_features=self.config['embed_dim'], out_features=output_size))
            self.pred_head_t = nn.Sequential(
                nn.Linear(in_features=self.config['embed_dim'], out_features=self.config['embed_dim']),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(in_features=self.config['embed_dim'], out_features=output_size))
            self.pred_head_union = nn.Sequential(
                nn.Linear(in_features=self.config['embed_dim'], out_features=self.config['embed_dim']),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(in_features=self.config['embed_dim'], out_features=output_size))
        elif self.fusion_strategy == 'residual':
            self.pred_head_residual = nn.Sequential(
                nn.Linear(in_features=4*self.config['embed_dim'], out_features=4*self.config['embed_dim']),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(in_features=4*self.config['embed_dim'], out_features=4*self.config['embed_dim']))
            self.output_layer = nn.Linear(in_features=4*self.config['embed_dim'], out_features=output_size)
            
    def forward(self, sentences, visual, visual_key_padding_mask, acoustic, audio_key_padding_mask, lengths, bert_sent, bert_sent_type, bert_sent_mask, targets, alpha=0, train=True):
        # project the textual/visual/audio features
        self.image_embed = self.vision_proj(visual) # self.image_embeds torch.Size([40, 64, 768])
        self.audio_embed = self.audio_proj(acoustic) # self.audio_embeds.shape torch.Size([40, 64, 768])       
        with torch.no_grad():    
            bert_output = self.pre_text_encoder(input_ids=bert_sent, attention_mask=bert_sent_mask, token_type_ids=bert_sent_type)
            # bert_output.keys() odict_keys(['last_hidden_state', 'pooler_output', 'hidden_states']) # pooler_output only corresponds to the first token
            # bert_output = bert_output[0]
            bert_output = bert_output['last_hidden_state'] # bert_output.shape torch.Size([8, 17, 768])
            # torch.stack(bert_output['hidden_states'][:-4]).shape torch.Size([4, 8, 17, 768])
            # bert_output = torch.sum(torch.stack(bert_output['hidden_states'][-4:]), dim=0) 
            # bert_output = F.normalize(bert_output, dim=-1) !!! do not use normalize
            masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)        
            # bert output should have dimension [batch_size, seq_len, n_features]
            self.text_embeds = masked_output.permute(1,0,2) # self.text_embed.shape torch.Size([40,64,768])
            # self.text_embeds = bert_output.permute(1,0,2) # self.text_embed.shape torch.Size([40,64,768])
        
        # project the textual/visual/audio features
        # self.image_embed = self.proj_v(visual.transpose(1,2)).permute(0,2,1)   # self.image_embeds torch.Size([40, 64, 768])
        # self.audio_embed = self.proj_a(acoustic.transpose(1,2)).permute(0,2,1) # self.audio_embeds.shape torch.Size([40, 64, 768])
        # self.text_embeds = self.proj_t (self.text_embeds.transpose(1,2)).permute(0,2,1)
        
        output_v, output_a, output_t, shared_neck = self.multimodal_transformer_encoder(self.image_embed, visual_key_padding_mask, 
                                                                                        self.audio_embed, audio_key_padding_mask, 
                                                                                        self.text_embeds, bert_sent_mask)
        
        # output_v.shape torch.Size([1, 8, 768])
        # output_a.shape torch.Size([1, 8, 768])
        # output_t.shape torch.Size([1, 8, 768])
        # shared_neck.shape torch.Size([1, 12, 768])
        
        # previous version
        # union_embeds = torch.cat((output_v, output_a, output_t, shared_neck),dim=0) # torch.Size([40+40+40+12, 64, 768])
        # encoder_output_mean = torch.mean(union_embeds, dim=0)
        # if train:
        #     prediction = self.fusion_head(encoder_output_mean)
        #     loss = self.criterion(prediction, targets)
        #     return loss
        # else:
        #     prediction = self.fusion_head(encoder_output_mean)
        #     return prediction
        # current version
        # We can also consider the output sequnences [batch, max_len, embedding_dim], the average across max_len dimensions to get averaged/mean embeddings
        # Step 1: Expand Attention/Padding Mask from [batch_size, max_len] to [batch_size, max_len, hidden_size].
        # Step 2: Sum Embeddings along max_len axis so now we have [batch_size, hidden_size].
        # Step 3: Sum Mask along max_len axis. This is done so that we can ignore padding tokens.
        # Step 4: Take Average.
        vision_mask_expanded = (~visual_key_padding_mask).float().unsqueeze(-1).expand(output_v.size())
        text_mask_expanded = bert_sent_mask.unsqueeze(-1).expand(output_t.size()).float()
        audio_mask_expanded = (~audio_key_padding_mask).float().unsqueeze(-1).expand(output_a.size())
        vision_sum_embeddings = torch.sum(output_v * vision_mask_expanded, axis=1)
        text_sum_embeddings = torch.sum(output_t * text_mask_expanded, axis=1)
        audio_sum_embeddings = torch.sum(output_a * audio_mask_expanded, axis=1)
        
        vision_sum_mask = vision_mask_expanded.sum(axis=1)
        vision_sum_mask = torch.clamp(vision_sum_mask, min=1e-9)
        text_sum_mask = text_mask_expanded.sum(axis=1)
        text_sum_mask = torch.clamp(text_sum_mask, min=1e-9)
        audio_sum_mask = audio_mask_expanded.sum(axis=1)
        audio_sum_mask = torch.clamp(audio_sum_mask, min=1e-9)

        vision_mean_embeddings = vision_sum_embeddings / vision_sum_mask # torch.Size([1, 768]
        text_mean_embeddings = text_sum_embeddings / text_sum_mask # torch.Size([1, 768]
        audio_mean_embeddings = audio_sum_embeddings / audio_sum_mask # torch.Size([1, 768]
        pivotal_mean_embeddings= torch.mean(shared_neck, axis=1)

        if train:
            if self.fusion_strategy == 'separate':
                pred_v = self.pred_head_v(vision_mean_embeddings)
                pred_a = self.pred_head_a(audio_mean_embeddings)
                pred_t = self.pred_head_t(text_mean_embeddings)
                pred_pivotal = self.pred_head_union(pivotal_mean_embeddings)
                if self.config['loss_type'] == 'mse':
                    loss_v = self.criterion(pred_v, targets) + self.criterion2(torch.clamp(pred_v, min=-3, max=3), torch.clamp(targets, min=-3, max=3))
                    loss_a = self.criterion(pred_a, targets) + self.criterion2(torch.clamp(pred_a, min=-3, max=3), torch.clamp(targets, min=-3, max=3))
                    loss_t = self.criterion(pred_t, targets) + self.criterion2(torch.clamp(pred_t, min=-3, max=3), torch.clamp(targets, min=-3, max=3))
                    loss_pivotal = self.criterion(pred_pivotal, targets) + self.criterion2(torch.clamp(pred_pivotal, min=-3, max=3), torch.clamp(targets, min=-3, max=3))
                elif self.config['loss_type'] == 'bce':
                    loss_v = self.criterion(pred_v, targets)
                    loss_a = self.criterion(pred_a, targets)
                    loss_t = self.criterion(pred_t, targets)
                    loss_pivotal = self.criterion(pred_pivotal, targets)
                # todo
                # prediction = (pred_v + pred_a + pred_t + pred_pivotal) /4.0  
                # loss = self.criterion(prediction, targets)
                loss = loss_v + loss_a + loss_t + loss_pivotal
                return loss
            elif self.fusion_strategy == 'residual':
                union_embeds = torch.cat([vision_mean_embeddings, text_mean_embeddings, audio_mean_embeddings, pivotal_mean_embeddings], dim=-1)
                output_embeds = self.pred_head_residual(union_embeds)
                # residual module 
                final_embeds = union_embeds + output_embeds 
                pred = self.output_layer(final_embeds)
                loss  = self.criterion(pred, targets)
                return loss
        else:
            if self.fusion_strategy == 'separate':
                pred_v = self.pred_head_v(vision_mean_embeddings)
                pred_a = self.pred_head_a(audio_mean_embeddings)
                pred_t = self.pred_head_t(text_mean_embeddings)
                pred_pivotal = self.pred_head_union(pivotal_mean_embeddings)
                prediction = (pred_v + pred_a + pred_t + pred_pivotal) / 4.0 # equal contribution
                return prediction
            elif self.fusion_strategy == 'residual':
                union_embeds = torch.cat([vision_mean_embeddings, text_mean_embeddings, audio_mean_embeddings, pivotal_mean_embeddings], dim=-1)
                output_embeds = self.pred_head_residual(union_embeds)
                # residual module
                final_embeds = union_embeds + output_embeds
                prediction = self.output_layer(final_embeds)
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