import copy
import torch
import random
import numpy as np
from torch import nn
from torch import einsum
from functools import partial
import torch.nn.functional as F
from typing import Optional, Any
from einops import rearrange, repeat

from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from transformers import BertModel, BertConfig
from torch.nn.modules.container import ModuleList
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.activation import MultiheadAttention
from models.vit import VisionTransformer, interpolate_pos_embed

import speechbrain as sb
from speechbrain.lobes.models.transformer.Transformer import PositionalEncoding, RelPosEncXL
from models.position_embedding import SinusoidalPositionalEmbedding, PositionalEncoding1D

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# positional
def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')

class AcFormer(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,    
                 temp = 0.07,
                 init_vision = False,
                 init_audio = False
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.mlm_probability = config['mlm_probability']
        embed_dim = config['embed_dim']
        vision_width = config['vision_width']
        audio_width = config['audio_width']

        if init_vision: # set false default
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)
        if init_audio:
            checkpoint =  ''
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.audio_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.audio_encoder.load_state_dict(state_dict,strict=False)
            print(msg)

        visual_encoder_layer = nn.TransformerEncoderLayer(d_model=vision_width, nhead=1)
        self.visual_encoder = nn.TransformerEncoder(visual_encoder_layer, num_layers=4)

        audio_encoder_layer = nn.TransformerEncoderLayer(d_model=config['audio_width'], nhead=1)
        self.audio_encoder = nn.TransformerEncoder(audio_encoder_layer, num_layers=4)

        bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.audio_proj = nn.Linear(audio_width, embed_dim)
        
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.audio_proj = nn.Linear(audio_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)    

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])   
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']  
        self.pred_head = nn.Linear(vision_width+audio_width+text_width, 2)

        # create momentum models
        visual_encoder_layer_m = nn.TransformerEncoderLayer(d_model=vision_width, nhead=1)
        self.visual_encoder_m = nn.TransformerEncoder(visual_encoder_layer_m, num_layers=4)

        audio_encoder_layer_m = nn.TransformerEncoderLayer(d_model=audio_width, nhead=1)
        self.audio_encoder_m = nn.TransformerEncoder(audio_encoder_layer_m, num_layers=4)

        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.audio_proj_m = nn.Linear(audio_width, embed_dim)
        
        self.text_encoder_m = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)     
        self.text_proj_m = nn.Linear(text_width, embed_dim)   

        self.model_pairs = [
                [self.visual_encoder,self.visual_encoder_m],
                [self.vision_proj,self.vision_proj_m],
                [self.audio_encoder,self.audio_encoder_m],
                [self.audio_proj,self.audio_proj_m],
                [self.text_encoder,self.text_encoder_m],
                [self.text_proj,self.text_proj_m],]
        
        self.copy_params()
 
        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("audio_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.audio_queue = nn.functional.normalize(self.audio_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
 
    def forward(self, image, image_aug, audio, audio_aug, bert_sent, bert_sent_type, bert_sent_mask, alpha=0):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)

        self.image_embeds = self.visual_encoder(image) # image_embeds.shape torch.Size(torch.Size([17, 8, 47]))
        image_embeds = self.image_embeds.permute(1, 0, 2)
        self.audio_embeds = self.audio_encoder(audio)  # audio_embeds.shape torch.Size([17, 8, 74])
        audio_embeds = self.audio_embeds.permute(1, 0, 2)

        bert_output = self.text_encoder(input_ids=bert_sent, attention_mask=bert_sent_mask, token_type_ids=bert_sent_type)
        bert_output = bert_output[0]
        masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)        
        text_embeds = masked_output # text_embeds.shape torch.Size([8, 17, 768])
      
        # image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)                     
        self.image_feat = F.normalize(self.vision_proj(image_embeds), dim=-1)
        image_feat = torch.mean(self.image_feat, dim=1)
        self.audio_feat  = F.normalize(self.audio_proj(audio_embeds), dim=-1)
        audio_feat = torch.mean(self.audio_feat, dim=1)
        self.text_feat = F.normalize(self.text_proj(text_embeds), dim=-1)
        text_feat = torch.mean(self.text_feat, dim=1)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            self.image_embeds_m = self.visual_encoder_m(image_aug)
            image_embeds_m  = self.image_embeds_m.permute(1, 0, 2) 
            self.audio_embeds_m = self.audio_encoder_m(audio_aug)
            audio_embeds_m = self.audio_embeds_m.permute(1, 0, 2)  
            bert_output = self.text_encoder_m(input_ids=bert_sent, attention_mask=bert_sent_mask, token_type_ids=bert_sent_type)
            bert_output = bert_output[0] 
            masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)        
            text_embeds_m = masked_output

            # acformer: local features of visual part
            # image_feat_m_l = self.patch_pooling(image_feat_m_l) # pooling for image patches
            image_feat_m_l = F.normalize(self.vision_proj_m(image_embeds_m),dim=-1)
            image_feat_m = torch.mean(image_feat_m_l, dim=1)
            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)             

            # audio_feat_m_l = self.patch_pooling(audio_feat_m_l) # pooling for spectralgram patches
            audio_feat_m_l = F.normalize(self.audio_proj_m(audio_embeds_m),dim=-1)
            audio_feat_m = torch.mean(audio_feat_m_l, dim=1)
            audio_feat_all = torch.cat([audio_feat_m.t(),self.audio_queue.clone().detach()],dim=1)             

            text_feat_m_l = F.normalize(self.text_proj_m(text_embeds_m),dim=-1)
            text_feat_m = torch.mean(text_feat_m_l, dim = 1)
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)
            
            sim_i2a_m = image_feat_m @ audio_feat_all / self.temp 
            sim_a2i_m = audio_feat_m @ image_feat_all / self.temp     

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp 
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp     

            sim_a2t_m = audio_feat_m @ text_feat_all / self.temp
            sim_t2a_m = text_feat_m @ audio_feat_all / self.temp

            sim_targets_ia = torch.zeros(sim_i2a_m.size()).to(image.device)
            sim_targets_ia.fill_diagonal_(1)
            sim_i2a_targets = alpha * F.softmax(sim_i2a_m, dim=1) + (1 - alpha) * sim_targets_ia
            sim_a2i_targets = alpha * F.softmax(sim_a2i_m, dim=1) + (1 - alpha) * sim_targets_ia     

            sim_targets_it = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets_it.fill_diagonal_(1)
            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets_it
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets_it     

            sim_targets_at = torch.zeros(sim_a2t_m.size()).to(image.device)
            sim_targets_at.fill_diagonal_(1)
            sim_a2t_targets = alpha * F.softmax(sim_a2t_m, dim=1) + (1 - alpha) * sim_targets_at
            sim_t2a_targets = alpha * F.softmax(sim_t2a_m, dim=1) + (1 - alpha) * sim_targets_at
        
        sim_i2a = image_feat @ audio_feat_all / self.temp 
        sim_a2i = audio_feat @ image_feat_all / self.temp

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp 

        sim_a2t = audio_feat @ text_feat_all / self.temp
        sim_t2a = text_feat @ audio_feat_all / self.temp

        loss_i2a = -torch.sum(F.log_softmax(sim_i2a, dim=1)*sim_i2a_targets,dim=1).mean()
        loss_a2i = -torch.sum(F.log_softmax(sim_a2i, dim=1)*sim_a2i_targets,dim=1).mean() 

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

        loss_a2t = -torch.sum(F.log_softmax(sim_a2t, dim=1)*sim_a2t_targets,dim=1).mean()
        loss_t2a = -torch.sum(F.log_softmax(sim_t2a, dim=1)*sim_t2a_targets,dim=1).mean() 

        # acformer: add inMod g2l loss
        loss_i2i_inmod_l = self.in_batch_g2l_loss(image_feat_m_l, image_feat, self.temp)
        loss_a2a_inmod_l = self.in_batch_g2l_loss(audio_feat_m_l, audio_feat, self.temp)
        loss_t2t_inmod_l = self.in_batch_g2l_loss(text_feat_m_l, text_feat, self.temp)

        # acformer: add in-modality g2g loss
        sim_i2i = image_feat @ image_feat_all / self.temp
        sim_a2a = audio_feat @ audio_feat_all / self.temp
        sim_t2t = text_feat @ text_feat_all / self.temp

        loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1)*sim_targets_ia,dim=1).mean()
        loss_a2a = -torch.sum(F.log_softmax(sim_a2a, dim=1)*sim_targets_it,dim=1).mean()
        loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1)*sim_targets_at,dim=1).mean()
        # compute multimodal vision-audio-text loss
        loss_vat = (loss_i2i_inmod_l + loss_a2a_inmod_l + loss_t2t_inmod_l  + loss_i2a + loss_a2i + loss_i2t + loss_t2i + loss_a2t + loss_t2a + loss_i2i + loss_a2a + loss_t2t) / 12.0
        
        self._dequeue_and_enqueue(image_feat_m, audio_feat_m, text_feat_m)
        
        ######===================================================================#########
        # forward the positve image-audio-text triplet
        bs = image_embeds.shape[0]
        # select a negative image for each image-audio-text triplet
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.randint(0, bs, (1,)).item()
            while neg_idx == b:
                neg_idx = torch.randint(0, bs, (1,)).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)                                     

        # select a negative audio for each image-audio-text triplet
        audio_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.randint(0, bs, (1,)).item()
            while neg_idx == b:
                neg_idx = torch.randint(0, bs, (1,)).item()
            audio_embeds_neg.append(audio_embeds[neg_idx])
        audio_embeds_neg = torch.stack(audio_embeds_neg, dim=0)                                     

        # select a negative text for each image-audio-text triplet
        text_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.randint(0, bs, (1,)).item()
            while neg_idx == b:
                neg_idx = torch.randint(0, bs, (1,)).item()
            text_embeds_neg.append(text_embeds[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0) 

        # multimodal image-audio-text triplet loss
        # (neg_image,pos_audio, pos_text) (pos_image, neg_audio, pos_text) (pos_image, pos_audio, neg_text)
        # image_embeds_neg.shape torch.Size([8, 17, 47]) image_embeds.shape torch.Size([8, 17, 47])
        self.image_embeds_all = torch.cat([image_embeds_neg, image_embeds, image_embeds],dim=0) # self.image_embeds_all.shape torch.Size([24, 17, 47])
        # audio_embeds.shape torch.Size([8, 17, 74]) audio_embeds_neg.shape  torch.Size([8, 17, 74])
        self.audio_embeds_all = torch.cat([audio_embeds, audio_embeds_neg, audio_embeds],dim=0) # self.audio_embeds_all.shape torch.Size([24, 17, 74])
        # text_embeds.shape torch.Size([8, 17, 768]) text_embeds_neg.shape torch.Size([8, 17, 768])
        self.text_embeds_all = torch.cat([text_embeds, text_embeds, text_embeds_neg],dim=0) # self.text_embeds_all.shape torch.Size([24, 17, 768])

        vat_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(3*bs,dtype=torch.long)],dim=0).to(image.device) # torch.Size([32])
        pos_vat_embeds = torch.mean(torch.cat([image_embeds, audio_embeds, text_embeds],dim=-1), dim=1) # torch.Size([8, 889])
        neg_vat_embeds = torch.mean(torch.cat([self.image_embeds_all, self.audio_embeds_all, self.text_embeds_all],dim=-1), dim=1) # torch.Size([24, 889])
        vat_embeddings = torch.cat([pos_vat_embeds, neg_vat_embeds], dim=0)  # torch.Size([32, 889])
        
        vat_output = self.pred_head(vat_embeddings)
        loss_vat_m = F.cross_entropy(vat_output, vat_labels)

        return loss_vat, loss_vat_m  

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
                
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, audio_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        audio_feats = concat_all_gather(audio_feat)
        text_feats = concat_all_gather(text_feat)
 
        batch_size = image_feats.shape[0]
 
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity
 
        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.audio_queue[:, ptr:ptr + batch_size] = audio_feats.T        
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
 
        self.queue_ptr[0] = ptr 
        
    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            
 
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id
 
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    # jinyu: patch pooling of image patches to reduce computation and enlarge receptive field
    def patch_pooling(self, x):
        batch_size, seq_length, dim = x.size()
        b1 = int(np.sqrt(seq_length))
        x = x.reshape(batch_size, b1, b1, dim)
        x = x.permute(0,3,1,2)
        c1 = int(np.sqrt(b1))
        x = F.avg_pool2d(x, c1, stride=c1)
        x = x.permute(0,2,3,1).reshape(batch_size, c1*c1, dim)
        return x

    # jinyu: in-batch g2l loss
    def in_batch_g2l_loss(self, l, m, temp, attention_mask=None):
        m = m.unsqueeze(1)
        N, n_locals, dim = l.size()
        l_n = l.reshape(-1, dim) # (N * n_locals) * d
        m_n = m.reshape(-1, dim) # N * d

        # Inner product for positive samples. Outer product for negative. We need to do it this way
        # for the multiclass loss. For the outer product, we want a N x N x n_locals x 1 tensor.
        u_p = torch.matmul(l, m.permute(0,2,1)).unsqueeze(2) / temp # N * n_locals * 1 * 1
        
        # if l comes from text, then attention_mask is not None
        if attention_mask is not None:
            temp_mask = attention_mask.unsqueeze(2).unsqueeze(3)
            u_p = (temp_mask * u_p) + (10000. * (1-temp_mask))
        
        u_n = torch.mm(m_n, l_n.t()) / temp
        u_n = u_n.reshape(N, 1, N, n_locals).permute(0, 2, 3, 1) # N x N x n_locals x 1

        # We need to mask the diagonal part of the negative tensor.
        mask = torch.eye(N)[:, :, None, None].to(l.device) # N*N*1*1
        n_mask = 1 - mask

        # Masking is done by shifting the diagonal before exp.
        u_n = (n_mask * u_n) - (10000. * (1 - n_mask))  # mask out "self" examples
        # if l comes from test, we mask out the padding tokens
        if attention_mask is not None:
            temp_mask = attention_mask.unsqueeze(0).unsqueeze(3).expand(N, -1, -1, -1)
            u_n = (temp_mask * u_n) - (10000. * (1-temp_mask))

        u_n = u_n.reshape(N, N * n_locals, 1).unsqueeze(dim=1).expand(-1, n_locals, -1, -1)

        # Since this is multiclass, we concat the positive along the class dimension before performing log softmax.
        pred_lgt = torch.cat([u_p, u_n], dim=2)
        pred_log = F.log_softmax(pred_lgt, dim=2)

        # The positive score is the first element of the log softmax.
        if attention_mask is not None:
            loss = (torch.sum(-pred_log[:, :, 0].squeeze(), dim=1) / torch.sum(attention_mask, dim=1)).mean()
        else:
            loss = -pred_log[:, :, 0].mean()

        return loss

# modules
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.heads = num_heads
        head_dim = dim // self.heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        q = q * self.scale

        attn = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')

        return self.proj_drop(self.proj(x))

class VideoTokenizer(nn.Module):
    def __init__(
        self,
        frame_kernel_size,
        kernel_size,
        stride,
        padding,
        frame_stride=1,
        frame_pooling_stride=1,
        frame_pooling_kernel_size=1,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        n_conv_layers=1,
        n_input_channels=3,
        n_output_channels=64,
        in_planes=64,
        activation=None,
        max_pool=True,
        conv_bias=False
    ):
        super().__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        n_filter_list_pairs = zip(n_filter_list[:-1], n_filter_list[1:])

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                # nn.Conv3d(chan_in, chan_out,
                #           kernel_size=(frame_kernel_size, kernel_size, kernel_size),
                #           stride=(frame_stride, stride, stride),
                #           padding=(frame_kernel_size // 2, padding, padding), bias=conv_bias), # !!! padding problems
                nn.Conv3d(chan_in, chan_out,
                          kernel_size=(frame_kernel_size,kernel_size,kernel_size),
                          stride=(frame_kernel_size,kernel_size,kernel_size)),                         
                nn.Identity() if not exists(activation) else activation(),
                nn.MaxPool3d(kernel_size=(frame_pooling_kernel_size, pooling_kernel_size, pooling_kernel_size),
                             stride=(frame_pooling_stride, pooling_stride, pooling_stride),
                             padding=(frame_pooling_kernel_size // 2, pooling_padding, pooling_padding)) if max_pool else nn.Identity()
            )
                for chan_in, chan_out in n_filter_list_pairs
            ])

        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, frames=8, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, frames, height, width))).shape[1]

    def forward(self, x):
        x = self.conv_layers(x)
        return rearrange(x, 'b c f h w -> b (f h w) c')

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        batch, drop_prob, device, dtype = x.shape[0], self.drop_prob, x.device, x.dtype

        if drop_prob <= 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (batch, *((1,) * (x.ndim - 1)))

        keep_mask = torch.zeros(shape, device = device).float().uniform_(0, 1) < keep_prob
        output = x.div(keep_prob) * keep_mask.float()
        return output

class VideoTransformerEncoderLayer(nn.Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and rwightman's timm package.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super().__init__()

        self.pre_norm = nn.LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1  = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1    = nn.LayerNorm(d_model)
        self.linear2  = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate)

        self.activation = F.gelu

    def forward(self, src, *args, **kwargs):
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src

class VideoTransformer(nn.Module):
    def __init__(
        self,
        seq_pool=True,
        embedding_dim=768,
        num_layers=6,
        num_heads=4,
        mlp_ratio=4.0,
        dropout_rate=0.1,
        attention_dropout=0.1,
        stochastic_depth_rate=0.1,
        positional_embedding='sine',
        sequence_length=None,
        tokenizer=None,
        *args, **kwargs):
        
        super().__init__()
        assert positional_embedding in {'sine', 'learnable', 'none'}

        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool
        self.tokenizer = tokenizer

        assert exists(sequence_length) or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        # if not seq_pool:
        #     sequence_length += 1
        #     self.class_emb = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        # else:
        #     self.attention_pool = nn.Linear(self.embedding_dim, 1)

        if positional_embedding == 'none':
            self.positional_emb = None
        elif positional_embedding == 'learnable':
            self.positional_emb = nn.Parameter(torch.zeros(1, sequence_length, embedding_dim))
            nn.init.trunc_normal_(self.positional_emb, std = 0.2)
        else:
            self.register_buffer('positional_emb', sinusoidal_embedding(sequence_length, embedding_dim))

        self.dropout = nn.Dropout(p=dropout_rate)

        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]

        self.blocks = nn.ModuleList([
            VideoTransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout_rate,
                                    attention_dropout=attention_dropout, drop_path_rate=layer_dpr)
            for layer_dpr in dpr])

        self.norm = nn.LayerNorm(embedding_dim)
        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and exists(m.bias):
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        b = x.shape[0]
        x = self.tokenizer(x)
        if not exists(self.positional_emb) and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)
        if exists(self.positional_emb):
            x += self.positional_emb
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # if self.seq_pool:
        #     attn_weights = rearrange(self.attention_pool(x), 'b n 1 -> b n')
        #     x = einsum('b n, b n d -> b d', attn_weights.softmax(dim = 1), x)
        # else:
        #     x = x[:, 0]
        return x

class AudioTransformer(nn.Module):
        def __init__(
        self,
        d_model=768,
        nhead=8,
        num_encoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        positional_encoding=None,
        normalize_before=True,
        encoder_module: Optional[str] = "transformer",
        attention_type: Optional[str] = "regularMHA",
        max_length: Optional[int] = 2500,
        causal: Optional[bool] = False,
        encoder_kdim: Optional[int] = None,
        encoder_vdim: Optional[int] = None,):
            super().__init__()
            self.causal = causal
            self.attention_type = attention_type
            self.positional_encoding_type = positional_encoding
            self.encoder_kdim = encoder_kdim
            self.encoder_vdim = encoder_vdim

            assert attention_type in ["regularMHA", "RelPosMHAXL"]
            assert positional_encoding in ["fixed_abs_sine", None]
            assert (
                num_encoder_layers> 0
            ), "number of encoder layers must larger than zero"

            if positional_encoding == "fixed_abs_sine":
                # self.positional_encoding = PositionalEncoding(d_model, max_length)
                self.positional_encoding = PositionalEncoding1D(d_model)

            elif positional_encoding is None:
                pass
                # no positional encodings

            # overrides any other pos_embedding
            if attention_type == "RelPosMHAXL":
                self.positional_encoding = RelPosEncXL(d_model)

            # initialize the encoder
            if num_encoder_layers > 0:
                if encoder_module == "transformer":
                    self.encoder = sb.lobes.models.transformer.Transformer.TransformerEncoder(
                        nhead=nhead,
                        num_layers=num_encoder_layers,
                        d_ffn=d_ffn,
                        d_model=d_model,
                        dropout=dropout,
                        activation=activation,
                        normalize_before=normalize_before,
                        causal=self.causal,
                        attention_type=self.attention_type,
                        kdim=self.encoder_kdim,
                        vdim=self.encoder_vdim,
                    )
            else:
                raise NotImplementedError("Not implement Encoder")
    
        def forward(self, audio_input, audio_key_padding_mask=None):
            """Users should modify this function according to their own tasks.                
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the position
            with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
            S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.                        
            """
            B, time_frames, freq_bins = audio_input.shape # audio_input.shape>>> torch.Size([1, 1024, 768])            
            # pos_encodings = self.positional_encoding(audio_input) # pos_embeds.shape>>> torch.Size([1, 1024, 768])
            # audio_key_padding_mask.shape torch.Size([1, 1024])
            # audio_input += pos_encodings                        
            # e.g. audio_key_padding_mask  tensor([[False, False, False,  ...,  True,  True,  True]], device='cuda:0')
            # True Ignore False Unchanged
            output, attention_lst = self.encoder(src = audio_input, 
                                                 src_mask = None,
                                                 src_key_padding_mask = audio_key_padding_mask, 
                                                 pos_embs=None)
            return output
        
class AcFormerPretrain(nn.Module):
    def __init__(self, config):
        super(AcFormerPretrain, self).__init__()
        self.config = config
        self.vision_width = vision_width = config['vision_width']
        self.audio_width = audio_width = config['audio_width']
        self.text_width = text_width = config['text_width']
        self.embed_dim = embed_dim = config['embed_dim']
        self.pre_fusion_layer = pre_fusion_layer = config['pre_fusion_layer']
        self.img_size = config['image_size']
        self.num_frames = config['num_frames']
        self.positional_embedding = config['positional_embedding']
        self.pre_text_encoder = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        ############################################# set unimodal VideoEncoder #############################################
        # batch_size first
        self.video_tokenizer = VideoTokenizer(
            n_input_channels=3,
            n_output_channels=self.vision_width,
            frame_stride=1,
            frame_kernel_size=2, # tube_size
            frame_pooling_stride=1,
            frame_pooling_kernel_size=1,
            kernel_size=16, # patch_size
            stride=2,
            padding=3,
            pooling_kernel_size=3,
            pooling_stride=2,
            pooling_padding=1,
            max_pool=True,
            activation=nn.ReLU,
            n_conv_layers=1,
            conv_bias=False)
        img_height, img_width = pair(self.img_size)
        video_sequence_length = self.video_tokenizer.sequence_length(n_channels=3, frames=self.num_frames, height=img_height, width=img_width)
        self.visual_encoder = VideoTransformer(
            sequence_length=video_sequence_length,
            embedding_dim=self.vision_width,
            seq_pool=True,
            dropout_rate=0.,
            attention_dropout=0.1,
            stochastic_depth_rate=0.1,
            positional_embedding=self.positional_embedding,
            tokenizer=self.video_tokenizer,
            num_heads = 1,
            mlp_ratio = 2.0,
            num_layers = self.pre_fusion_layer)        
        ############################################# set unimodal AudioEncoder #############################################
        # batch_size first
        self.audio_encoder = AudioTransformer(
            d_model = self.audio_width,
            nhead = 1,
            num_encoder_layers = self.pre_fusion_layer,
            d_ffn = int(2*self.audio_width),
            encoder_module = 'transformer',
            attention_type = "regularMHA",
            positional_encoding = None)
        ############################################# set unimodal TextEncoder #############################################
        # batch_size second
        UNI_text_encoder_layer = TransformerEncoderLayer(d_model=self.text_width, nhead=1)
        self.text_encoder = TransformerEncoder(UNI_text_encoder_layer, pre_fusion_layer, nn.LayerNorm(text_width))
        
        # contrastive projection 
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.audio_proj = nn.Linear(audio_width, embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])   
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']  
        self.pred_head = nn.Linear(embed_dim*3, 2)

        # create momentum models
        self.visual_encoder_m = VideoTransformer(
            sequence_length=video_sequence_length,
            embedding_dim=self.vision_width,
            seq_pool=True,
            dropout_rate=0.,
            attention_dropout=0.1,
            stochastic_depth_rate=0.1,
            positional_embedding=self.positional_embedding,
            tokenizer=self.video_tokenizer,
            num_heads = 1,
            mlp_ratio = 2.0,
            num_layers = self.pre_fusion_layer)
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        
        self.audio_encoder_m = AudioTransformer(
            d_model = self.audio_width,
            nhead = 1,
            num_encoder_layers = self.pre_fusion_layer,
            d_ffn = int(2*self.audio_width),
            encoder_module = 'transformer',
            attention_type = "regularMHA",
            positional_encoding = None)
        self.audio_proj_m = nn.Linear(audio_width, embed_dim)

        self.text_encoder_m =  TransformerEncoder(UNI_text_encoder_layer, pre_fusion_layer, nn.LayerNorm(text_width))        
        self.text_proj_m = nn.Linear(text_width, embed_dim)   
        
        self.model_pairs = [
                [self.visual_encoder,self.visual_encoder_m],
                [self.vision_proj,self.vision_proj_m],
                [self.audio_encoder,self.audio_encoder_m],
                [self.audio_proj,self.audio_proj_m],
                [self.text_encoder,self.text_encoder_m],
                [self.text_proj,self.text_proj_m],]
        
        
        self.copy_params()
        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("audio_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.audio_queue = nn.functional.normalize(self.audio_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    def forward(self, video_input, video_input_aug, audio_input, audio_key_padding_mask, bert_sent, bert_sent_type, bert_sent_mask, alpha=0, is_train=True):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
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
        
        image_embeds = self.visual_encoder(video_input) # image_embeds.shape torch.Size([batchsize, 5408, vision_width])
        # image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)
        image_feat = F.normalize(self.vision_proj(torch.mean(image_embeds, dim=1)), dim=-1)
        audio_embeds = self.audio_encoder(audio_input, audio_key_padding_mask)
        # sb_transformer >> text_output.shape torch.Size([batch_size, time_frames, audio_width])        
        text_output = self.text_encoder(self.text_embeds, src_key_padding_mask = ~bert_sent_mask.bool())
        text_embeds = text_output.permute(1,0,2)
        # text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)  
        audio_mask_expanded = (~audio_key_padding_mask).float().unsqueeze(-1).expand(audio_embeds.size())
        text_mask_expanded = bert_sent_mask.unsqueeze(-1).expand(text_embeds.size()).float()
        text_sum_embeddings = torch.sum(text_embeds * text_mask_expanded, axis=1)
        audio_sum_embeddings = torch.sum(audio_embeds * audio_mask_expanded, axis=1)
        text_sum_mask = text_mask_expanded.sum(axis=1)
        text_sum_mask = torch.clamp(text_sum_mask, min=1e-9)
        audio_sum_mask = audio_mask_expanded.sum(axis=1)
        audio_sum_mask = torch.clamp(audio_sum_mask, min=1e-9)
        mean_audio_feat = audio_sum_embeddings / audio_sum_mask # torch.Size([1, audio_width]
        mean_text_feat = text_sum_embeddings / text_sum_mask # torch.Size([1, text_width]
        audio_feat = F.normalize(self.audio_proj(mean_audio_feat),dim=-1)
        text_feat = F.normalize(self.text_proj(mean_text_feat),dim=-1)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(video_input) 
            image_feat_m = F.normalize(self.vision_proj_m(torch.mean(image_embeds_m,dim=1)),dim=-1)
            # local features of visual part
            image_feat_m_l = F.normalize(self.vision_proj_m(image_embeds_m),dim=-1)  
            # image_feat_m_l = self.patch_pooling(image_feat_m_l) # pooling for image patches
            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)

            audio_embeds_m = self.audio_encoder_m(audio_input, audio_key_padding_mask)
            # sb_transformer >> text_output.shape torch.Size([batch_size, time_frames, audio_width])        
            text_output_m = self.text_encoder_m(self.text_embeds, src_key_padding_mask = ~bert_sent_mask.bool())
            text_embeds_m = text_output_m.permute(1,0,2)
            # text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)  
            
            audio_mask_expanded_m = (~audio_key_padding_mask).float().unsqueeze(-1).expand(audio_embeds_m.size())
            text_mask_expanded_m = bert_sent_mask.unsqueeze(-1).expand(text_embeds_m.size()).float()
            text_sum_embeddings_m = torch.sum(text_embeds_m * text_mask_expanded_m, axis=1)
            audio_sum_embeddings_m = torch.sum(audio_embeds_m * audio_mask_expanded_m, axis=1)
            text_sum_mask_m = text_mask_expanded_m.sum(axis=1)
            text_sum_mask_m = torch.clamp(text_sum_mask_m, min=1e-9)
            audio_sum_mask_m = audio_mask_expanded_m.sum(axis=1)
            audio_sum_mask_m = torch.clamp(audio_sum_mask_m, min=1e-9)
            mean_audio_feat_m = audio_sum_embeddings_m / audio_sum_mask_m # torch.Size([1, audio_width]
            mean_text_feat_m = text_sum_embeddings_m / text_sum_mask_m # torch.Size([1, text_width]

            audio_feat_m = F.normalize(self.audio_proj_m(mean_audio_feat_m),dim=-1)
            audio_feat_m_l = F.normalize(self.audio_proj_m(audio_embeds_m), dim=-1)
            audio_feat_all = torch.cat([audio_feat_m.t(),self.audio_queue.clone().detach()],dim=1)
 
            # text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1)
            text_feat_m = F.normalize(self.text_proj_m(mean_text_feat_m),dim=-1)
            # text_feat_m_l = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,1:,:]),dim=-1) 
            text_feat_m_l = F.normalize(self.text_proj_m(text_embeds_m),dim=-1) 
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

            sim_i2a_m = image_feat_m @ audio_feat_all / self.temp 
            sim_a2i_m = audio_feat_m @ image_feat_all / self.temp     

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp 
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp     

            sim_a2t_m = audio_feat_m @ text_feat_all / self.temp
            sim_t2a_m = text_feat_m @ audio_feat_all / self.temp

            sim_targets_ia = torch.zeros(sim_i2a_m.size()).to(video_input.device)
            sim_targets_ia.fill_diagonal_(1)
            sim_i2a_targets = alpha * F.softmax(sim_i2a_m, dim=1) + (1 - alpha) * sim_targets_ia
            sim_a2i_targets = alpha * F.softmax(sim_a2i_m, dim=1) + (1 - alpha) * sim_targets_ia     

            sim_targets_it = torch.zeros(sim_i2t_m.size()).to(video_input.device)
            sim_targets_it.fill_diagonal_(1)
            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets_it
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets_it     

            sim_targets_at = torch.zeros(sim_a2t_m.size()).to(video_input.device)
            sim_targets_at.fill_diagonal_(1)
            sim_a2t_targets = alpha * F.softmax(sim_a2t_m, dim=1) + (1 - alpha) * sim_targets_at
            sim_t2a_targets = alpha * F.softmax(sim_t2a_m, dim=1) + (1 - alpha) * sim_targets_at
        
        sim_i2a = image_feat @ audio_feat_all / self.temp 
        sim_a2i = audio_feat @ image_feat_all / self.temp

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp 

        sim_a2t = audio_feat @ text_feat_all / self.temp
        sim_t2a = text_feat @ audio_feat_all / self.temp

        loss_i2a = -torch.sum(F.log_softmax(sim_i2a, dim=1)*sim_i2a_targets,dim=1).mean()
        loss_a2i = -torch.sum(F.log_softmax(sim_a2i, dim=1)*sim_a2i_targets,dim=1).mean() 

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

        loss_a2t = -torch.sum(F.log_softmax(sim_a2t, dim=1)*sim_a2t_targets,dim=1).mean()
        loss_t2a = -torch.sum(F.log_softmax(sim_t2a, dim=1)*sim_t2a_targets,dim=1).mean() 

        # acformer: add inMod g2l loss
        loss_i2i_inmod_l = self.in_batch_g2l_loss(image_feat_m_l, image_feat, self.temp)
        loss_a2a_inmod_l = self.in_batch_g2l_loss(audio_feat_m_l, audio_feat, self.temp)
        loss_t2t_inmod_l = self.in_batch_g2l_loss(text_feat_m_l, text_feat, self.temp)

        # acformer: add in-modality g2g loss
        sim_i2i = image_feat @ image_feat_all / self.temp
        sim_a2a = audio_feat @ audio_feat_all / self.temp
        sim_t2t = text_feat @ text_feat_all / self.temp

        loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1)*sim_targets_ia,dim=1).mean()
        loss_a2a = -torch.sum(F.log_softmax(sim_a2a, dim=1)*sim_targets_it,dim=1).mean()
        loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1)*sim_targets_at,dim=1).mean()
        # compute multimodal vision-audio-text loss
        loss_vat = (loss_i2i_inmod_l + loss_a2a_inmod_l + loss_t2t_inmod_l  + loss_i2a + loss_a2i + loss_i2t + loss_t2i + loss_a2t + loss_t2a + loss_i2i + loss_a2a + loss_t2t) / 12.0
        
        self._dequeue_and_enqueue(image_feat_m, audio_feat_m, text_feat_m)
        
        ######===================================================================#########
        # forward the positve image-audio-text triplet
        bs = image_feat.shape[0]
        # select a negative image for each image-audio-text triplet
        image_feats_neg = []
        for b in range(bs):
            neg_idx = torch.randint(0, bs, (1,)).item()
            while neg_idx == b:
                neg_idx = torch.randint(0, bs, (1,)).item()
            image_feats_neg.append(image_feat[neg_idx])
        image_feats_neg = torch.stack(image_feats_neg, dim=0)                                     

        # select a negative audio for each image-audio-text triplet
        audio_feats_neg = []
        for b in range(bs):
            neg_idx = torch.randint(0, bs, (1,)).item()
            while neg_idx == b:
                neg_idx = torch.randint(0, bs, (1,)).item()
            audio_feats_neg.append(audio_feat[neg_idx])
        audio_feats_neg = torch.stack(audio_feats_neg, dim=0)                                     

        # select a negative text for each image-audio-text triplet
        text_feats_neg = []
        for b in range(bs):
            neg_idx = torch.randint(0, bs, (1,)).item()
            while neg_idx == b:
                neg_idx = torch.randint(0, bs, (1,)).item()
            text_feats_neg.append(text_feat[neg_idx])
        text_feats_neg = torch.stack(text_feats_neg, dim=0) 

        # multimodal image-audio-text triplet loss
        # (neg_image,pos_audio, pos_text) (pos_image, neg_audio, pos_text) (pos_image, pos_audio, neg_text)
        # image_embeds_neg.shape torch.Size([2, 5408, 256]) image_embeds.shape torch.Size([2, 5408, 256])
        # audio_embeds.shape torch.Size([2, 267, 128]) audio_embeds_neg.shape  torch.Size([2, 267, 128])
        # text_embeds.shape torch.Size([2, 9, 768] text_embeds_neg.shape torch.Size([2, 9, 768]
        
        self.image_feats_all = torch.cat([image_feats_neg, image_feat, image_feat],dim=0) 
        self.audio_feats_all = torch.cat([audio_feat, audio_feats_neg, audio_feat],dim=0) 
        self.text_feats_all = torch.cat([text_feat, text_feat, text_feats_neg],dim=0)
        
        vat_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(3*bs,dtype=torch.long)],dim=0).to(video_input.device) 
        pos_vat_embeds = torch.cat([image_feat, audio_feat, text_feat],dim=-1)
        neg_vat_embeds = torch.cat([self.image_feats_all, self.audio_feats_all, self.text_feats_all],dim=-1)
        vat_embeddings = torch.cat([pos_vat_embeds, neg_vat_embeds], dim=0)
        
        vat_output = self.pred_head(vat_embeddings)
        if is_train:
            loss_vat_m = F.cross_entropy(vat_output, vat_labels, reduction='mean')
            return loss_vat, loss_vat_m, loss_i2a, loss_a2i, loss_i2t, loss_t2i, loss_a2t, loss_t2a, loss_i2i, loss_a2a, loss_t2t
        else:
            return image_feat, audio_feat, text_feat

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
                
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, audio_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        audio_feats = concat_all_gather(audio_feat)
        text_feats = concat_all_gather(text_feat)
 
        batch_size = image_feats.shape[0]
 
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity
 
        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.audio_queue[:, ptr:ptr + batch_size] = audio_feats.T        
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
 
        self.queue_ptr[0] = ptr 
        
    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            
 
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id
 
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    # jinyu: patch pooling of image patches to reduce computation and enlarge receptive field
    def patch_pooling(self, x):
        batch_size, seq_length, dim = x.size()
        b1 = int(np.sqrt(seq_length))
        x = x.reshape(batch_size, b1, b1, dim)
        x = x.permute(0,3,1,2)
        c1 = int(np.sqrt(b1))
        x = F.avg_pool2d(x, c1, stride=c1)
        x = x.permute(0,2,3,1).reshape(batch_size, c1*c1, dim)
        return x

    # jinyu: in-batch g2l loss
    def in_batch_g2l_loss(self, l, m, temp, attention_mask=None):
        m = m.unsqueeze(1)
        N, n_locals, dim = l.size()
        l_n = l.reshape(-1, dim) # (N * n_locals) * d
        m_n = m.reshape(-1, dim) # N * d

        # Inner product for positive samples. Outer product for negative. We need to do it this way
        # for the multiclass loss. For the outer product, we want a N x N x n_locals x 1 tensor.
        u_p = torch.matmul(l, m.permute(0,2,1)).unsqueeze(2) / temp # N * n_locals * 1 * 1
        
        # if l comes from text, then attention_mask is not None
        if attention_mask is not None:
            temp_mask = attention_mask.unsqueeze(2).unsqueeze(3)
            u_p = (temp_mask * u_p) + (10000. * (1-temp_mask))
        
        u_n = torch.mm(m_n, l_n.t()) / temp
        u_n = u_n.reshape(N, 1, N, n_locals).permute(0, 2, 3, 1) # N x N x n_locals x 1

        # We need to mask the diagonal part of the negative tensor.
        mask = torch.eye(N)[:, :, None, None].to(l.device) # N*N*1*1
        n_mask = 1 - mask

        # Masking is done by shifting the diagonal before exp.
        u_n = (n_mask * u_n) - (10000. * (1 - n_mask))  # mask out "self" examples
        # if l comes from test, we mask out the padding tokens
        if attention_mask is not None:
            temp_mask = attention_mask.unsqueeze(0).unsqueeze(3).expand(N, -1, -1, -1)
            u_n = (temp_mask * u_n) - (10000. * (1-temp_mask))

        u_n = u_n.reshape(N, N * n_locals, 1).unsqueeze(dim=1).expand(-1, n_locals, -1, -1)

        # Since this is multiclass, we concat the positive along the class dimension before performing log softmax.
        pred_lgt = torch.cat([u_p, u_n], dim=2)
        pred_log = F.log_softmax(pred_lgt, dim=2)

        # The positive score is the first element of the log softmax.
        if attention_mask is not None:
            loss = (torch.sum(-pred_log[:, :, 0].squeeze(), dim=1) / torch.sum(attention_mask, dim=1)).mean()
        else:
            loss = -pred_log[:, :, 0].mean()

        return loss

class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)        
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

            Shape:
                see the docs in Transformer class.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the position
            with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
            S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
 
    output = torch.cat(tensors_gather, dim=0)
    return output
