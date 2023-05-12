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

class AttentionBottleneckTransformer(Module):
    """Attention Bottleneck Transformer.
    This class implements one of advanced methods to explore multiple modalities separately
    but with sharing information between them.
    See also: `Attention Bottlenecks for Multimodal Fusion`_.
    .. _`Attention Bottlenecks for Multimodal Fusion`: https://arxiv.org/pdf/2107.00135.pdf
    """
    def __init__(self, 
                 UNI_image_encoder_layer, UNI_audio_encoder_layer, UNI_text_encoder_layer, 
                 MUL_image_encoder_layer, MUL_audio_encoder_layer, MUL_text_encoder_layer, 
                 num_layers, fusion_layer, neck_size, embed_dim, norm=None):
        super(AttentionBottleneckTransformer, self).__init__()
        self.num_layers = num_layers
        self.fusion_layer = fusion_layer # starting fusion layer
        self.mbt_layer = num_layers - fusion_layer 
        assert self.fusion_layer>=0 and self.fusion_layer<= self.num_layers-1, "check your fusion layer"
        
        self.unimodal_vision_layers = _get_clones(UNI_image_encoder_layer, self.fusion_layer)
        self.unimodal_audio_layers = _get_clones(UNI_audio_encoder_layer, self.fusion_layer)
        self.unimodal_text_layers = _get_clones(UNI_text_encoder_layer, self.fusion_layer)

        self.multimodal_vision_layers = _get_clones(MUL_image_encoder_layer, self.mbt_layer)
        self.multimodal_aduio_layers = _get_clones(MUL_audio_encoder_layer, self.mbt_layer)
        self.multimodal_text_layers = _get_clones(MUL_text_encoder_layer, self.mbt_layer)

        self.bottleneck = nn.Parameter(data=torch.zeros(neck_size,1, embed_dim))
        self.neck_size = neck_size
        self.norm = LayerNorm(embed_dim)

    def forward(self, src_v, src_a, src_t, 
                mask_v=None, mask_a=None, mask_t=None,
                src_key_padding_mask_v=None, src_key_padding_mask_a=None, src_key_padding_mask_t=None):
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
        output_v = src_v
        output_a = src_a
        output_t = src_t

        assert src_v.shape[1] == src_a.shape[1] and src_a.shape[1]==src_t.shape[1], "batch size error: check your modality input"
        batch_size = src_v.shape[1]
        vison_seq_len = src_v.shape[0]
        audio_seq_len = src_a.shape[0]
        text_seq_len = src_t.shape[0]
        shared_neck = self.bottleneck.expand(-1, batch_size, -1) # torch.Size([12, 8, 768])
        
        # unimodal encoders
        for mod in self.unimodal_vision_layers:
            output_v = mod(output_v, src_mask=mask_v, src_key_padding_mask=src_key_padding_mask_v)
        for mod in self.unimodal_audio_layers:
            output_a = mod(output_a, src_mask=mask_a, src_key_padding_mask=src_key_padding_mask_a)
        for mod in self.unimodal_text_layers:
            output_t = mod(output_t, src_mask=mask_t, src_key_padding_mask=src_key_padding_mask_t)
        
        # multimodal encoders
        for mod_v, mod_a, mod_t in zip(self.multimodal_aduio_layers, self.multimodal_aduio_layers, self.multimodal_text_layers):
            vison_neck = torch.cat((output_v,shared_neck),dim=0)
            audio_neck = torch.cat((output_a,shared_neck),dim=0)
            text_neck = torch.cat((output_t,shared_neck),dim=0)
            output_v = mod_v(vison_neck)[:vison_seq_len,:,:]
            z_fsn_v = mod_v(vison_neck)[vison_seq_len:,:,:]
            output_a = mod_a(audio_neck)[:audio_seq_len,:,:]
            z_fsn_a = mod_a(audio_neck)[audio_seq_len:,:,:]
            output_t = mod_t(text_neck)[:text_seq_len,:,:]
            z_fsn_t = mod_t(text_neck)[text_seq_len:,:,:]
            # z^{l+1}_{fsn} = Average_{i}(\hat{z}^{l+1}_{fsn_{i}})
            # shared_neck = reduce(torch.add, [z_fsn_v,z_fsn_a,z_fsn_t])
            shared_neck = torch.sum(torch.stack([z_fsn_v,z_fsn_a,z_fsn_t]), dim=0) / 3.0

        if self.norm is not None:
            output_v = self.norm(output_v)
            output_a = self.norm(output_a)
            output_t = self.norm(output_t)
            shared_neck = self.norm(shared_neck)
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



class AttentionBottleneckTransformer_(nn.Module):
    """Attention Bottleneck Transformer.
    This class implements one of advanced methods to explore multiple modalities separately
    but with sharing information between them.
    See also: `Attention Bottlenecks for Multimodal Fusion`_.
    .. _`Attention Bottlenecks for Multimodal Fusion`: https://arxiv.org/pdf/2107.00135.pdf
    """

    def __init__(
        self,
        transformers: list[dict[str, Any]] | tuple[dict[str, Any]],
        embed_dims: int = 768,
        neck_size: int = 4,
        cls_only: bool = True,
        **kwargs,
    ):
        """
        Args:
            transformers (list[dict[str, Any]] | tuple[dict[str, Any]]): list of
                transformer configs.
            embed_dims (int, optional): size of embedding.
                Defaults to 768.
            neck_size (int, optional): size of bottleneck which is shared between transformers.
                Defaults to 4.
            cls_only (bool, optional): boolean flag to return only class token.
                Otherwise the full tensor with class token and features is returned.
                Defaults to True.
        """
        super().__init__()
        assert isinstance(transformers, list), "check your transformers"
        
        self.transformers = transformers
        self.bottleneck = nn.Parameter(data=torch.zeros(1, neck_size, embed_dims))
        self.cls_only = cls_only

    def forward(self, *per_transformer_x):
        """Forwards input tensors and shared bottleneck to correpsonding transformers.
        It also calculates and stores bottleneck for next iteration
        as mean of bottlenecks after each transformer.
        Each transformer takes same shared bottleneck.
        Args:
            per_transformer_x (tuple[Tensor, ...]): tuple of batch inputs for corresponding transformers.
        Returns:
            tuple[Tensor, ...] | Tensor: if cls_only is True when it returns tuple of tensors of features
            with class token of after each transformer. Otherwise only class token is returned.
        """
        batch_size = per_transformer_x[0].size(0)
        shared_neck = self.bottleneck.expand(batch_size, -1, -1)
        next_shared_neck = torch.zeros(shared_neck.size())

        for x, transformer in zip(per_transformer_x, self.transformers):
            x = torch.cat((x, shared_neck), dim=1)
            x = transformer(x)
            next_shared_neck += x[:, -shared_neck.size(1) :]

            if self.cls_only:
                x = x[:, 0]
            else:
                x = x[:, : -shared_neck.size(1)]

        next_shared_neck /= len(per_transformer_x)
        self.bottleneck.copy_(next_shared_neck)

        return per_transformer_x
    


def logging_names(func):
    
    def wrapper():
        logging.warn("%s is running" % func.__name__)
        return func()
    return wrapper



def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

