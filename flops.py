from thop import profile, clever_format

import copy
import torch
import logging
import torch.nn as nn
from thop import profile
from torch import Tensor
from torch import nn, einsum

import torch.nn.functional as F
from einops import rearrange, repeat

from functools import reduce
from torch.nn.init import xavier_uniform_

from typing import Optional
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.container import ModuleList
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.activation import MultiheadAttention

import pickle
import numpy as np
from tqdm import tqdm_notebook
import hashlib

def get_word_list_sdk(path_to_embedding, word_list_emb_sdk, embedding_size=300, embedding_vocab=2196017, init_emb=None):
    f = open(path_to_embedding, 'r')
    found = 0
    big_emb_dict = dict()
    for index, line in enumerate(tqdm_notebook(f, total=embedding_vocab)):
        content = line.strip().split()
        word = ' '.join(content[:-300])
        vector = np.asarray(list(map(lambda x: float(x), content[-300:])))
        m = hashlib.md5(np.float32(vector))
        # m = hashlib.md5(np.float32(vector).astype("uint8"))
        hash_str = m.hexdigest()
        if big_emb_dict.get(hash_str, '') != '':
            debug = 1
        big_emb_dict[hash_str] = word
        if index % 1000 == 0:
            print(index)

    word_list = dict()
    for j_idx, word_list_emb in enumerate(word_list_emb_sdk):
        # m = hashlib.md5(np.float32(word_list_emb).astype("uint8"))
        m = hashlib.md5(np.float32(word_list_emb))
        hash_str = m.hexdigest()
        word = big_emb_dict.get(hash_str, '')
        word_list[j_idx] = word
    
    with open('/word_list.pkl', 'wb') as f:
        pickle.dump(word_list, f)

#     cosine_smi_matrix = np.array(cosine_smi_matrix).transpose(0, 1)
#     return word_list_sdk  # len: 24311


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


class SimpleMultimodalAttention(Module):
    """
    This class implements a multimodal fusion scheme via vanilla self-attention, that is, all modality inputs are concatenated into a single sequence, which is then fed to several standard transformer layers with vanilla  self-attention block. 
    See more details in `Attention Is All You Need.` 
    arXiv preprint arXiv:1706.03762 (2017).
    "
    """
    def __init__(self, fusion_layer, embed_dim, fusion_head):
        super(SimpleMultimodalAttention, self).__init__()
        self.fusion_layer = fusion_layer
        self.embed_dim = embed_dim
        self.fusion_head = fusion_head
        
        MUL_encoder_layer = TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)
        self.multimodal_layers = _get_clones(MUL_encoder_layer, self.fusion_layer)

        self.norm = nn.LayerNorm(embed_dim)
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

    def forward(self, video_input, audio_input, text_input):
        """
        (L, N, E) :: L is the target sequence length, N is the batch size, E is the embedding dimension
        """
        batch_size = video_input.shape[1]
        assert all(input.shape[1] == batch_size for input in [audio_input, text_input])
       
        vat_uni_output = torch.cat([video_input, audio_input, text_input], dim=0)
        for mod_uni in self.multimodal_layers:
            vat_uni_output = mod_uni(vat_uni_output)       
        
        if self.norm is not None:
            vat_uni_output = self.norm(vat_uni_output)
        return vat_uni_output
    

class CrossAttention(Module):
    """
    This class implements a co-attentional transformer, where the module computes query, key, and value matrices as in a standard transformer block. 
    And the keys and values from each modality are passed as input to the other modality’s multi-headed attention block.
    See details in
    Lu, Jiasen, et al. "ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks." arXiv preprint arXiv:1908.02265 (2019).
    """
    def __init__(self, fusion_layer, embed_dim, fusion_head):
        super(CrossAttention, self).__init__()
        self.fusion_layer = fusion_layer
        self.embed_dim = embed_dim
        self.fusion_head = fusion_head

        MUL_visual_encoder_layer = TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)
        MUL_audio_encoder_layer = TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)
        MUL_text_encoder_layer = TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)

        # multimodal encoders
        self.multimodal_vision_layers = _get_clones(MUL_visual_encoder_layer, self.fusion_layer)
        self.multimodal_aduio_layers = _get_clones(MUL_audio_encoder_layer, self.fusion_layer)
        self.multimodal_text_layers = _get_clones(MUL_text_encoder_layer, self.fusion_layer)

        self.norm = nn.LayerNorm(embed_dim)
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

    def forward(self, video_input, audio_input, text_input):
        batch_size = video_input.shape[1]
        assert all(input.shape[1] == batch_size for input in [audio_input, text_input]) 
        output_v = video_input
        output_a = audio_input
        output_t = text_input

        for mod_v, mod_a, mod_t in zip(self.multimodal_vision_layers, 
                                       self.multimodal_aduio_layers, 
                                       self.multimodal_text_layers):
            output_v_a = mod_v(output_v, output_a)
            output_v_t = mod_v(output_v, output_t)
            output_a_v = mod_a(output_a, output_v)
            output_a_t = mod_a(output_a, output_t)
            output_t_v = mod_t(output_t, output_v)
            output_t_a = mod_t(output_t, output_a)
            output_v = (output_v_a + output_v_t) / 2.0 
            output_a = (output_a_v + output_a_t) / 2.0 
            output_t = (output_t_a + output_t_v) / 2.0 
        if self.norm is not None:
            output_v = self.norm(output_v)
            output_a = self.norm(output_a)
            output_t = self.norm(output_t)

        return output_v, output_a, output_t

class BottleneckAttention(Module):
    """
    This class implements a transformer basedarchitecture that uses 'fusion bottlenecks' for multimodal fusion and restricts information between different modalities to pass through 
    a small number of bottleneck latents.
    See also: `Attention Bottlenecks for Multimodal Fusion`_.
    .. _`Attention Bottlenecks for Multimodal Fusion`: https://arxiv.org/pdf/2107.00135.pdf
    """
    def __init__(self, fusion_layer, neck_size, embed_dim, fusion_head):
        super(BottleneckAttention, self).__init__()
        self.fusion_layer = fusion_layer
        self.neck_size = neck_size
        self.embed_dim = embed_dim
        self.fusion_head = fusion_head

        MUL_visual_encoder_layer = TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)
        MUL_audio_encoder_layer = TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)
        MUL_text_encoder_layer = TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)

        # multimodal encoders
        self.multimodal_vision_layers = _get_clones(MUL_visual_encoder_layer, self.fusion_layer)
        self.multimodal_aduio_layers = _get_clones(MUL_audio_encoder_layer, self.fusion_layer)
        self.multimodal_text_layers = _get_clones(MUL_text_encoder_layer, self.fusion_layer)

        self.bottleneck = nn.Parameter(data=torch.zeros(neck_size, 1, embed_dim))
        self.neck_size = neck_size
        self.norm = nn.LayerNorm(embed_dim)
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

    def forward(self, video_input, audio_input,  text_input):
        """
        (L, N, E) :: L is the target sequence length, N is the batch size, E is the embedding dimension
        """
        batch_size = video_input.shape[1]
        assert all(input.shape[1] == batch_size for input in [audio_input, text_input]) 
        shared_neck = self.bottleneck.expand(-1, batch_size, -1) # torch.Size([12, 8, 768])

        vision_seq_len = video_input.shape[0]
        audio_seq_len = audio_input.shape[0]
        text_seq_len = text_input.shape[0]

        output_v = video_input
        output_a = audio_input
        output_t = text_input

        # multimodal encoder
        for mod_v, mod_a, mod_t in zip(self.multimodal_vision_layers, self.multimodal_aduio_layers, self.multimodal_text_layers):
            vison_neck = torch.cat((output_v,shared_neck), dim=0)
            audio_neck = torch.cat((output_a,shared_neck), dim=0)
            text_neck = torch.cat((output_t,shared_neck), dim=0)
            output_v = mod_v(vison_neck)[:vision_seq_len,:,:]
            z_fsn_v = mod_v(vison_neck)[vision_seq_len:,:,:]
            output_a = mod_a(audio_neck)[:audio_seq_len,:,:]
            z_fsn_a = mod_a(audio_neck)[audio_seq_len:,:,:]
            output_t = mod_t(text_neck)[:text_seq_len,:,:]
            z_fsn_t = mod_t(text_neck)[text_seq_len:,:,:]
            shared_neck = (z_fsn_v + z_fsn_a + z_fsn_t) / 3.0

        if self.norm is not None:
            output_v = self.norm(output_v)
            output_a = self.norm(output_a)
            output_t = self.norm(output_t)
            shared_neck = self.norm(shared_neck)
        return output_v, output_a, output_t, shared_neck


class PivotalAttention(Module):
    """Attention Bottleneck Transformer.
    This class implements one of advanced methods to explore multiple modalities separately
    but with sharing information between them.
    See also: `Attention Bottlenecks for Multimodal Fusion`_.
    .. _`Attention Bottlenecks for Multimodal Fusion`: https://arxiv.org/pdf/2107.00135.pdf
    """
    def __init__(self, fusion_layer, neck_size, embed_dim, fusion_head):
        super(PivotalAttention, self).__init__()
        self.fusion_layer = fusion_layer
        self.embed_dim = embed_dim
        self.neck_size = neck_size
        self.fusion_head = fusion_head

        # pivotal attention bottleneck
        TOKEN_collection_vision_layer = PivotalTransformerDecoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)
        TOKEN_collection_audio_layer = PivotalTransformerDecoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)
        TOKEN_collection_text_layer = PivotalTransformerDecoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)
        TOKEN_propagation_vision_layer = PivotalTransformerDecoderLayer_v2(d_model=self.embed_dim, nhead=self.fusion_head)
        TOKEN_propagation_audio_layer = PivotalTransformerDecoderLayer_v2(d_model=self.embed_dim, nhead=self.fusion_head)
        TOKEN_propagation_text_layer = PivotalTransformerDecoderLayer_v2(d_model=self.embed_dim, nhead=self.fusion_head)

        # collection layers
        self.token_collection_vision_layers = _get_clones(TOKEN_collection_vision_layer, self.fusion_layer)
        self.token_collection_audio_layers = _get_clones(TOKEN_collection_audio_layer, self.fusion_layer)
        self.token_collection_text_layers = _get_clones(TOKEN_collection_text_layer, self.fusion_layer)

        # propagation layers
        self.token_propagation_vision_layers = _get_clones(TOKEN_propagation_vision_layer, self.fusion_layer)
        self.token_propagation_audio_layers = _get_clones(TOKEN_propagation_audio_layer, self.fusion_layer)
        self.token_propagation_text_layers = _get_clones(TOKEN_propagation_text_layer, self.fusion_layer)

        self.bottleneck = nn.Parameter(data=torch.zeros(neck_size, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        
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

    def forward(self, video_input, audio_input, text_input):
        """
        :param orginal_video_input: the input frames of video, expected shape: (batch_size, channels, frames, height, width), e.g., (12, 3, 8, 224, 224)     
        :param orginal_audio_input: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :param orginal_audio_input: the inptu bert text embeddings, expected shape (seq_length, batch_size, embedding_dim), e.g., (18, 12, 768)
        :return: prediction
        """
        batch_size = video_input.shape[1]
        assert all(input.shape[1] == batch_size for input in [audio_input, text_input]) 
        shared_neck = self.bottleneck.expand(-1, batch_size, -1) # torch.Size([12, 8, 768])         
        # shared encoders
        
        output_v = video_input
        output_a = audio_input
        output_t = text_input

        for col_v, col_a, col_t, pro_v, pro_a, pro_t in zip(self.token_collection_vision_layers,
                                                            self.token_collection_audio_layers,
                                                            self.token_collection_text_layers,
                                                            self.token_propagation_vision_layers,
                                                            self.token_propagation_audio_layers,
                                                            self.token_propagation_text_layers):                                    
            neck_vision = col_v(shared_neck, output_v) # neck_vision.shape (neck_size,batch_size, embedding_dim)
            neck_audio = col_a(shared_neck, output_a)  # neck_audio.shape (neck_size, batch_size, embedding_dim)
            neck_text = col_t(shared_neck, output_t)   # neck_text.shape (neck_size, batch_size, embedding_dim)
            shared_neck += (neck_vision + neck_audio + neck_text) / 3.0
            output_v = pro_v(output_v, shared_neck)
            output_a = pro_a(output_a, shared_neck)
            output_t = pro_t(output_t, shared_neck)

        if self.norm is not None:
            output_v = self.norm(output_v)
            output_a = self.norm(output_a)
            output_t = self.norm(output_t)
            shared_neck = self.norm(shared_neck)
        
        return output_v, output_a, output_t, shared_neck


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
    

class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
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
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class PivotalTransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
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
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(PivotalTransformerDecoderLayer, self).__init__()
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(PivotalTransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        
        tgt = self.norm1(tgt)
        
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        
        tgt = self.norm2(tgt)
        
        return tgt

class PivotalTransformerDecoderLayer_v2(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
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
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(PivotalTransformerDecoderLayer_v2, self).__init__()
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(PivotalTransformerDecoderLayer_v2, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        
        tgt = self.norm1(tgt)
        
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt

def logging_names(func):
    
    def wrapper():
        logging.warn("%s is running" % func.__name__)
        return func()
    return wrapper

class GLU(Module):
    def __init__(self, embed_dim):
        super(GLU, self).__init__()
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        return self.linear1(x)*self.linear2(x).sigmoid()
    

if __name__ == '__main__':
    fusion_layer = 1
    neck_size = 12
    embed_dim = 768
    fusion_head = 8
    standard_attention_multimodal = SimpleMultimodalAttention(fusion_layer, embed_dim, fusion_head)
    cross_attention = CrossAttention(fusion_layer, embed_dim, fusion_head)
    bottleneck_attention = BottleneckAttention(fusion_layer, neck_size, embed_dim, fusion_head)
    pivotal_attention = PivotalAttention(fusion_layer, neck_size, embed_dim, fusion_head)
    
    # standard settings
    video_input = torch.randn(196,1,768)
    audio_input = torch.randn(98,1,768) # 1s audio fbank time frames
    text_input = torch.randn(256,1,768) # max bert input length 512

    macs, params = profile(standard_attention_multimodal, inputs=((video_input, audio_input, text_input)))
    macs, params = clever_format([macs, params], "%.3f")
    print("macs {0}, parmas {1} model {2}".format(macs, params, 'standard_attention_multimodal'))
    
    macs, params = profile(cross_attention, inputs=((video_input, audio_input, text_input)))
    macs, params = clever_format([macs, params], "%.3f")
    print("macs {0}, parmas {1} model {2}".format(macs, params, 'cross_attention'))

    macs, params = profile(bottleneck_attention, inputs=((video_input, audio_input, text_input)))
    macs, params = clever_format([macs, params], "%.3f")
    print("macs {0}, parmas {1} model {2}".format(macs, params, 'bottleneck_attention'))

    macs, params = profile(pivotal_attention, inputs=((video_input, audio_input, text_input)))
    macs, params = clever_format([macs, params], "%.3f")
    print("macs {0}, parmas {1} model {2}".format(macs, params, 'pivotal_attention'))
