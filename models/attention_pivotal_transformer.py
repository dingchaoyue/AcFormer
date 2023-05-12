from __future__ import annotations
import copy
import torch
import logging
import torch.nn as nn
from torch import Tensor
from functools import reduce
from torch.nn.init import xavier_uniform_

from typing import Optional, Any
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.container import ModuleList
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.activation import MultiheadAttention
from models.position_embedding import SinusoidalPositionalEmbedding, PositionalEncoding1D

class AttentionPivotalTransformer(Module):
    """Attention Bottleneck Transformer.
    This class implements one of advanced methods to explore multiple modalities separately
    but with sharing information between them.
    See also: `Attention Bottlenecks for Multimodal Fusion`_.
    .. _`Attention Bottlenecks for Multimodal Fusion`: https://arxiv.org/pdf/2107.00135.pdf
    """
    def __init__(self,
                 UNI_visual_encoder_layer, UNI_audio_encoder_layer, UNI_text_encoder_layer,
                 TOKEN_collection_vision_layer, TOKEN_collection_audio_layer, TOKEN_collection_text_layer,
                 TOKEN_propagation_vision_layer, TOKEN_propagation_audio_layer, TOKEN_propagation_text_layer,
                 num_layers, start_fusion_layer, neck_size, embed_dim, norm=None):
        super(AttentionPivotalTransformer, self).__init__()
        
        self.num_layers = num_layers
        self.start_fusion_layer = start_fusion_layer # starting fusion layer
        self.fusion_layer = num_layers - start_fusion_layer
        assert self.start_fusion_layer >= 0 and self.start_fusion_layer <= self.num_layers-1, "check your fusion layer"

        # self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        # self.embed_positions = PositionalEncoding1D(embed_dim)

        self.unimodal_vision_layers = _get_clones(UNI_visual_encoder_layer, self.start_fusion_layer)
        self.unimodal_audio_layers = _get_clones(UNI_audio_encoder_layer, self.start_fusion_layer)
        self.unimodal_text_layers = _get_clones(UNI_text_encoder_layer, self.start_fusion_layer)
        # collection layers
        self.token_collection_vision_layers = _get_clones(TOKEN_collection_vision_layer, self.fusion_layer)
        self.token_collection_audio_layers = _get_clones(TOKEN_collection_audio_layer, self.fusion_layer)
        self.token_collection_text_layers = _get_clones(TOKEN_collection_text_layer, self.fusion_layer)
        # propagation layers
        self.token_propagation_vision_layers = _get_clones(TOKEN_propagation_vision_layer, self.fusion_layer)
        self.token_propagation_audio_layers = _get_clones(TOKEN_propagation_audio_layer, self.fusion_layer)
        self.token_propagation_text_layers = _get_clones(TOKEN_propagation_text_layer, self.fusion_layer)

        self.bottleneck = nn.Parameter(data=torch.zeros(neck_size,1, embed_dim))
        self.neck_size = neck_size
        self.norm = LayerNorm(embed_dim)

    def forward(self, src_v, src_key_padding_mask_v, src_a, src_key_padding_mask_a, src_t, src_key_padding_mask_t):        
        """Pass the input through the encoder layers in turn.
        Args:
            src_v: the sequence to the vision encoder (required).
            src_a: the sequence to the audio encoder (required).
            src_t: the sequence to the text encoder (required).
            mask_v: the mask for the src_v sequence (optional).
            mask_a: the mask for the src_v sequence (optional).
            mask_t: the mask for the src_v sequence (optional).
            src_key_padding_mask_v: the mask for the src_v keys per batch (optional).
            src_key_padding_mask_a: the mask for the src_a keys per batch (optional).
            src_key_padding_mask_t: the mask for the src_t keys per batch (optional).
        Shape:
            src_v: (S,N,E), (S,B,E), namely batch_size second
            src_a: (S,N,E), (S,B,E), namely batch_size second
            src_t: (S,N,E), (S,B,E), namely batch_size second
        """
        batch_size = src_v.shape[1]
        assert all(input.shape[1] == batch_size for input in [src_a, src_t]), "batch size error: check your modality input"

        # borrow some codes from https://github.com/yaohungt/Multimodal-Transformer/blob/master/modules/transformer.py
        # output_v = src_v + self.embed_positions(src_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
        # output_a = src_a + self.embed_positions(src_a.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
        # output_t = src_t + self.embed_positions(src_t.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
        # output_v = src_v + self.embed_positions(src_v.transpose(0, 1)).transpose(0, 1)   # Add positional embedding
        # output_a = src_a + self.embed_positions(src_a.transpose(0, 1)).transpose(0, 1)   # Add positional embedding
        # output_t = src_t + self.embed_positions(src_t.transpose(0, 1)).transpose(0, 1)   # Add positional embedding
        output_v = src_v 
        output_a = src_a 
        output_t = src_t

        shared_neck = self.bottleneck.expand(-1, batch_size, -1) # torch.Size([12, 8, 768])
        visual_key_padding_mask = src_key_padding_mask_v
        audio_key_padding_mask = src_key_padding_mask_a
        text_key_padding_mask = ~src_key_padding_mask_t.bool()

        # unimodl encoders
        for mod in self.unimodal_vision_layers:
            output_v = mod(output_v, src_mask=None, src_key_padding_mask=visual_key_padding_mask)
        for mod in self.unimodal_audio_layers:
            output_a = mod(output_a, src_mask=None, src_key_padding_mask=audio_key_padding_mask)
        for mod in self.unimodal_text_layers:
            output_t = mod(output_t, src_mask=None, src_key_padding_mask=text_key_padding_mask)
        
        # shared encoders
        for col_v, col_a, col_t, pro_v, pro_a, pro_t in zip(self.token_collection_vision_layers,
                                                            self.token_collection_audio_layers,
                                                            self.token_collection_text_layers,
                                                            self.token_propagation_vision_layers,
                                                            self.token_propagation_audio_layers,
                                                            self.token_propagation_text_layers):
            neck_vision = col_v(shared_neck, output_v, memory_key_padding_mask=visual_key_padding_mask)
            neck_audio = col_a(shared_neck, output_a, memory_key_padding_mask=audio_key_padding_mask)
            neck_text = col_t(shared_neck, output_t, memory_key_padding_mask=text_key_padding_mask)
            # shared_neck = torch.sum(torch.stack([neck_vision, neck_audio, neck_text]), dim=0) / 3.0
            shared_neck = (neck_vision + neck_audio + neck_text) / 3.0
            output_v = pro_v(output_v, shared_neck,  tgt_key_padding_mask=visual_key_padding_mask)
            output_a = pro_a(output_a, shared_neck, tgt_key_padding_mask=audio_key_padding_mask)
            output_t = pro_t(output_t, shared_neck, tgt_key_padding_mask=text_key_padding_mask)

        if self.norm is not None:
            output_v = self.norm(output_v)
            output_a = self.norm(output_a)
            output_t = self.norm(output_t)
            shared_neck = self.norm(shared_neck)
        output_v, output_a, output_t, shared_neck = output_v.permute(1,0,2), output_a.permute(1,0,2), output_t.permute(1,0,2), shared_neck.permute(1,0,2)
        return output_v, output_a, output_t, shared_neck


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

    def forward(self, X):
        return self.linear1(X)*self.linear2(X).sigmoid()

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))