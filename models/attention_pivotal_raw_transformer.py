from __future__ import annotations
import copy
import torch
import logging
import torch.nn as nn
from torch import Tensor

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

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

from timm.models.layers import to_2tuple, trunc_normal_
from functools import partial

import speechbrain as sb
from speechbrain.lobes.models.transformer.Transformer import PositionalEncoding, RelPosEncXL


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


#  modules
class AudioMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AudioAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    def forward(self, x, register_hook=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AudioBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AudioAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = AudioMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, register_hook=False):
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


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
                nn.Conv3d(chan_in, chan_out,
                          kernel_size=(frame_kernel_size, kernel_size, kernel_size),
                          stride=(frame_stride, stride, stride),
                          padding=(frame_kernel_size // 2, padding, padding), bias=conv_bias),
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
        *args, **kwargs
    ):
    
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
    

class AudioPatchEmbed(nn.Module):
    def __init__(self, img_size=(128, 1024), patch_size = 16, stride_size=10, in_chans=1, embed_dim=768,
                norm_layer=None,flatten=True, bias=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size = to_2tuple(stride_size)
        # self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    @property
    def num_patches(self):
        test_input = torch.randn(1, 1, self.img_size[0],self.img_size[1])
        test_out = self.proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        num_patches = int(f_dim * t_dim)
        return num_patches

    def forward(self, x):
        B, C, H, W = x.shape
        assert  H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert  W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."    
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class AudioSpectrogramTransformer(nn.Module):
    """ Audio Spectrogram Transformer
    A PyTorch impl of : Gong, Yuan, Yu-An Chung, and James Glass. "Ast: Audio spectrogram transformer." arXiv preprint arXiv:2104.01778 (2021).
    """
    def __init__(self, audio_patch_embeder=None, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """   
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        assert isinstance(audio_patch_embeder, AudioPatchEmbed), "Input Audio PatchEmbed"
        self.patch_embed = audio_patch_embeder
        num_patches = self.patch_embed.num_patches
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule      
        self.blocks = nn.ModuleList([
            AudioBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, register_blk=-1):
        B, time_frames, freq_bins = x.shape
        x = x.unsqueeze(1)
        x = x.transpose(2, 3) # (input_fdim=128, input_tdim=1024) test_input=torch.randn(1, 1, input_fdim, input_tdim)         
        x = self.patch_embed(x) # x.shape BNC
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for i,blk in enumerate(self.blocks):
            x = blk(x, register_blk==i)
        x = self.norm(x)
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
        positional_encoding="fixed_abs_sine",
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
                self.positional_encoding = PositionalEncoding(d_model, max_length)
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
            pos_encodings = self.positional_encoding(audio_input) # pos_embeds.shape>>> torch.Size([1, 1024, 768])
            # audio_key_padding_mask.shape torch.Size([1, 1024])
            audio_input += pos_encodings                        
            # e.g. audio_key_padding_mask  tensor([[False, False, False,  ...,  True,  True,  True]], device='cuda:0')
            # True Ignore False Unchanged
            output, attention_lst = self.encoder(src = audio_input, 
                                                 src_mask = None,
                                                 src_key_padding_mask = audio_key_padding_mask, 
                                                 pos_embs=None)
            return output

class RawAttentionPivotalTransformer(Module):
    """Attention Bottleneck Transformer.
    This class implements one of advanced methods to explore multiple modalities separately
    but with sharing information between them.
    See also: `Attention Bottlenecks for Multimodal Fusion`_.
    .. _`Attention Bottlenecks for Multimodal Fusion`: https://arxiv.org/pdf/2107.00135.pdf
    
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    
    """
    def __init__(self, img_size, num_frames, freq_bins, time_frames, positional_embedding, num_layers, start_fusion_layer, neck_size, embed_dim, fusion_head, audio_model='transformer'):
        super(RawAttentionPivotalTransformer, self).__init__()
        
        self.img_size = img_size
        self.num_frames = num_frames
        self.num_layers = num_layers
        self.pre_fusion_layer = start_fusion_layer # starting fusion layer
        self.fusion_layer = num_layers - start_fusion_layer
        assert self.fusion_layer >= 0 and self.fusion_layer <= self.num_layers - 1, "check your fusion layer"
        self.embed_dim = embed_dim
        self.neck_size = neck_size
        self.fusion_head = fusion_head
        self.positional_embedding = positional_embedding
        self.input_fdim = freq_bins # 128
        self.input_tdim = time_frames # 1024
        self.fstride = 10 
        self.tstride = 10
        
        # set video unimodal encoder
        self.video_tokenizer = VideoTokenizer(
            n_input_channels=3,
            n_output_channels=self.embed_dim,
            frame_stride=1,
            frame_kernel_size=9,
            frame_pooling_stride=1,
            frame_pooling_kernel_size=1,
            kernel_size=14,
            stride=2,
            padding=3,
            pooling_kernel_size=9,
            pooling_stride=2,
            pooling_padding=1,
            max_pool=True,
            activation=nn.ReLU,
            n_conv_layers=1,
            conv_bias=False)
        
        img_height, img_width = pair(self.img_size)
        video_sequence_length = self.video_tokenizer.sequence_length(n_channels=3, frames=self.num_frames, height=img_height, width=img_width)        
        self.unimodal_vision_encoder = VideoTransformer(
            sequence_length=video_sequence_length,
            embedding_dim=self.embed_dim,
            seq_pool=True,
            dropout_rate=0.,
            attention_dropout=0.1,
            stochastic_depth_rate=0.1,
            positional_embedding=self.positional_embedding,
            tokenizer=self.video_tokenizer,
            num_heads = 1,
            mlp_ratio = 2.0,
            num_layers = self.pre_fusion_layer
            )        
        # set audio unimodal encoder
        if audio_model=='spectrogram':
            self.audio_patch_embeder = AudioPatchEmbed(
                img_size=(self.input_fdim, self.input_tdim), 
                patch_size= (16, 16),
                stride_size = (self.fstride, self.tstride),
                in_chans=1,
                embed_dim=self.embed_dim,
                )
            self.unimodal_audio_encoder = AudioSpectrogramTransformer(
                audio_patch_embeder=self.audio_patch_embeder, 
                embed_dim=self.embed_dim, 
                depth=self.pre_fusion_layer,
                num_heads= 1, 
                mlp_ratio= 2.0,
                )            
        elif audio_model=='transformer':
            """
            ----------------------------
            Example
            ----------------------------
            >>> import torch
            >>> x = torch.rand((8, 60, 512))
            >>> net = TransformerEncoder(1, 8, 512, d_model=512)
            >>> output, _ = net(x)
            >>> output.shape
            torch.Size([8, 60, 512])
            """            
            self.unimodal_audio_encoder = AudioTransformer(
                d_model = self.embed_dim,
                nhead = 1,
                num_encoder_layers = self.pre_fusion_layer,
                d_ffn = int(1*self.embed_dim),
                encoder_module = 'transformer',
                attention_type = "regularMHA",
                positional_encoding = "fixed_abs_sine",
                max_length = 1024,
                )

        # set text unimodal encoder
        UNI_text_encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=1)
        self.unimodal_text_encoder = TransformerEncoder(UNI_text_encoder_layer, self.pre_fusion_layer, norm= nn.LayerNorm(embed_dim))

        #  pivotal attention bottleneck
        TOKEN_collection_vision_layer = TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)
        TOKEN_collection_audio_layer = TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)
        TOKEN_collection_text_layer = TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)
        TOKEN_propagation_vision_layer = TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)
        TOKEN_propagation_audio_layer = TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)
        TOKEN_propagation_text_layer = TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)

        # collection layers
        self.token_collection_vision_layers = _get_clones(TOKEN_collection_vision_layer, self.fusion_layer)
        self.token_collection_audio_layers = _get_clones(TOKEN_collection_audio_layer, self.fusion_layer)
        self.token_collection_text_layers = _get_clones(TOKEN_collection_text_layer, self.fusion_layer)
        # propagation layers
        self.token_propagation_vision_layers = _get_clones(TOKEN_propagation_vision_layer, self.fusion_layer)
        self.token_propagation_audio_layers = _get_clones(TOKEN_propagation_audio_layer, self.fusion_layer)
        self.token_propagation_text_layers = _get_clones(TOKEN_propagation_text_layer, self.fusion_layer)

        self.bottleneck = nn.Parameter(data=torch.zeros(neck_size,1, embed_dim))
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

    def forward(self, video_input, audio_input, audio_key_padding_mask, text_input, text_mask):
        """
        :param orginal_video_input: the input frames of video, expected shape: (batch_size, channels, frames, height, width), e.g., (12, 3, 8, 224, 224)     
        :param orginal_audio_input: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :param orginal_audio_input: the inptu bert text embeddings, expected shape (seq_length, batch_size, embedding_dim), e.g., (18, 12, 768)
        :return: prediction
        """
        # expect audio input = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        assert (video_input.shape[0] == audio_input.shape[0]) and (video_input.shape[0] == text_input.shape[1]), "check your batch_size"
        B = video_input.shape[0]
        shared_neck = self.bottleneck.expand(-1, B, -1) # torch.Size([12, 8, 768])        
        # unimodl encoders
        ################################################################ Audio Mask ###############################################################
        output_v = self.unimodal_vision_encoder(video_input) # output_v.shape torch.Size([1, 12544, 768])
        # spectrogram >> output_a.shape torch.Size([1, 1212, 768])
        # transformer >> output_a.shape torch.Size([1, 1024, 768])
        output_a = self.unimodal_audio_encoder(audio_input, audio_key_padding_mask)  
        ################################################################ Text Mask ################################################################
        # True Ignore False Unchanged
        # text_mask.shape torch.Size([1, 10])   an example of text_mask
        # tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0')        
        # key_padding_mask shape (N,S), N::batch_size, S::sequence_length 1 or 0
        ################################################################ Text Mask ################################################################
        output_t = self.unimodal_text_encoder(text_input, src_key_padding_mask = ~text_mask.bool()) # output_t.shape torch.Size([10, 1, 768])
        # output_t = self.unimodal_text_encoder(text_input) # output_t.shape torch.Size([10, 1, 768])
        output_v = output_v.permute(1,0,2)
        output_a = output_a.permute(1,0,2)
        
        # shared encoders
        for col_v, col_a, col_t, pro_v, pro_a, pro_t in zip(self.token_collection_vision_layers,
                                                            self.token_collection_audio_layers,
                                                            self.token_collection_text_layers,
                                                            self.token_propagation_vision_layers,
                                                            self.token_propagation_audio_layers,
                                                            self.token_propagation_text_layers):
            
            neck_vision = col_v(shared_neck, output_v)
            neck_audio = col_a(shared_neck, output_a, memory_key_padding_mask = audio_key_padding_mask)
            neck_text = col_t(shared_neck, output_t, memory_key_padding_mask = ~text_mask.bool())
            # fusion operators
            shared_neck = torch.sum(torch.stack([neck_vision, neck_audio, neck_text]), dim=0) / 3.0
            output_v = pro_v(output_v, shared_neck)
            output_a = pro_a(output_a, shared_neck, tgt_key_padding_mask=audio_key_padding_mask)
            output_t = pro_t(output_t, shared_neck, tgt_key_padding_mask=~text_mask.bool())

        if self.norm is not None:
            output_v = self.norm(output_v)
            output_a = self.norm(output_a)
            output_t = self.norm(output_t)
            shared_neck = self.norm(shared_neck)
        output_v, output_a, output_t, shared_neck = output_v.permute(1,0,2), output_a.permute(1,0,2), output_t.permute(1,0,2), shared_neck.permute(1,0,2)
        return output_v, output_a, output_t, shared_neck


class RawPivotalTransformer(Module):
    """Attention Bottleneck Transformer.
    This class implements one of advanced methods to explore multiple modalities separately
    but with sharing information between them.
    See also: `Attention Bottlenecks for Multimodal Fusion`_.
    .. _`Attention Bottlenecks for Multimodal Fusion`: https://arxiv.org/pdf/2107.00135.pdf
    
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram    
    """
    def __init__(self, img_size, num_frames, freq_bins, time_frames, positional_embedding, num_layers, start_fusion_layer, neck_size, embed_dim, fusion_head, modality, audio_model='transformer'):
        super(RawPivotalTransformer, self).__init__()
        self.img_size = img_size
        self.num_frames = num_frames
        self.num_layers = num_layers
        self.pre_fusion_layer = start_fusion_layer # starting fusion layer
        self.fusion_layer = num_layers - start_fusion_layer
        assert self.fusion_layer >= 0 and self.fusion_layer <= self.num_layers - 1, "check your fusion layer"
        self.embed_dim = embed_dim
        self.neck_size = neck_size
        self.fusion_head = fusion_head
        self.positional_embedding = positional_embedding
        self.input_fdim = freq_bins # 128
        self.input_tdim = time_frames # 1024
        self.modality = modality
        assert modality in ['video', 'text', 'audio', 'video+audio', 'video+text', 'audio+text']
                  
        if self.modality in ['video','video+text', 'video+audio']:                
            # set video unimodal encoder
            self.video_tokenizer = VideoTokenizer(
                n_input_channels=3,
                n_output_channels=self.embed_dim,
                frame_stride=1,
                frame_kernel_size=9,
                frame_pooling_stride=1,
                frame_pooling_kernel_size=1,
                kernel_size=14,
                stride=2,
                padding=3,
                pooling_kernel_size=9,
                pooling_stride=2,
                pooling_padding=1,
                max_pool=True,
                activation=nn.ReLU,
                n_conv_layers=1,
                conv_bias=False)            
            img_height, img_width = pair(self.img_size)
            video_sequence_length = self.video_tokenizer.sequence_length(n_channels=3, frames=self.num_frames, height=img_height, width=img_width)        
            self.unimodal_vision_encoder = VideoTransformer(
                sequence_length=video_sequence_length,
                embedding_dim=self.embed_dim,
                seq_pool=True,
                dropout_rate=0.,
                attention_dropout=0.1,
                stochastic_depth_rate=0.1,
                positional_embedding=self.positional_embedding,
                tokenizer=self.video_tokenizer,
                num_heads = 1,
                mlp_ratio = 2.0,
                num_layers = self.pre_fusion_layer
                )         
        elif self.modality in ['audio', 'video+audio', 'audio+text']:
            # set audio unimodal encoder
            
            self.unimodal_audio_encoder = AudioTransformer(
                d_model = self.embed_dim,
                nhead = 1,
                num_encoder_layers = self.pre_fusion_layer,
                d_ffn = int(1*self.embed_dim),
                encoder_module = 'transformer',
                attention_type = "regularMHA",
                positional_encoding = "fixed_abs_sine",
                max_length = 1024,
                )
            
        elif self.modality in ['text', 'video+text', 'audio+text']:
            # set text unimodal encoder
            UNI_text_encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=1)
            self.unimodal_text_encoder = TransformerEncoder(UNI_text_encoder_layer, self.pre_fusion_layer)
        
        if self.modality == 'video+audio':
            # pivotal attention between two modalities
            self.bottleneck = nn.Parameter(data=torch.zeros(neck_size,1, embed_dim))
            TOKEN_collection_vision_layer = TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)
            TOKEN_propagation_vision_layer = TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)            
            TOKEN_collection_audio_layer = TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)
            TOKEN_propagation_audio_layer = TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)
            # collection layers
            self.token_collection_vision_layers = _get_clones(TOKEN_collection_vision_layer, self.fusion_layer)
            self.token_collection_audio_layers = _get_clones(TOKEN_collection_audio_layer, self.fusion_layer)
            # propagation layers
            self.token_propagation_vision_layers = _get_clones(TOKEN_propagation_vision_layer, self.fusion_layer)
            self.token_propagation_audio_layers = _get_clones(TOKEN_propagation_audio_layer, self.fusion_layer)
        
        elif self.modality == 'video+text':
            # pivotal attention between two modalities
            self.bottleneck = nn.Parameter(data=torch.zeros(neck_size,1, embed_dim))
            TOKEN_collection_vision_layer = TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)
            TOKEN_propagation_vision_layer = TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)
            TOKEN_collection_text_layer = TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)
            TOKEN_propagation_text_layer = TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)    
            # collection layers
            self.token_collection_vision_layers = _get_clones(TOKEN_collection_vision_layer, self.fusion_layer)
            self.token_collection_text_layers = _get_clones(TOKEN_collection_text_layer, self.fusion_layer)
            # propagation layers
            self.token_propagation_vision_layers = _get_clones(TOKEN_propagation_vision_layer, self.fusion_layer)
            self.token_propagation_text_layers = _get_clones(TOKEN_propagation_text_layer, self.fusion_layer)
        
        elif self.modality == 'audio+text':
            # pivotal attention between two modalities
            self.bottleneck = nn.Parameter(data=torch.zeros(neck_size,1, embed_dim))
            TOKEN_collection_audio_layer = TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)
            TOKEN_propagation_audio_layer = TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)
            TOKEN_collection_text_layer = TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)
            TOKEN_propagation_text_layer = TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.fusion_head)
            # collection layers
            self.token_collection_audio_layers = _get_clones(TOKEN_collection_audio_layer, self.fusion_layer)
            self.token_collection_text_layers = _get_clones(TOKEN_collection_text_layer, self.fusion_layer)
            # propagation layers
            self.token_propagation_audio_layers = _get_clones(TOKEN_propagation_audio_layer, self.fusion_layer)
            self.token_propagation_text_layers = _get_clones(TOKEN_propagation_text_layer, self.fusion_layer)

           
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

    def forward(self, video_input, audio_input, audio_key_padding_mask, text_input, text_mask):
        """
        :param orginal_video_input: the input frames of video, expected shape: (batch_size, channels, frames, height, width), e.g., (12, 3, 8, 224, 224)     
        :param orginal_audio_input: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :param orginal_audio_input: the inptu bert text embeddings, expected shape (seq_length, batch_size, embedding_dim), e.g., (18, 12, 768)
        :return: prediction
        """
        assert (video_input.shape[0] == audio_input.shape[0]) and (video_input.shape[0] == text_input.shape[1]), "check your batch_size"
        B = video_input.shape[0]

        if self.modality =='video':
            output_v = self.unimodal_vision_encoder(video_input) # output_v.shape torch.Size([1, 12544, 768])
            if self.norm is not None:
                output_v = self.norm(output_v)
            return output_v
        
        elif self.modality == 'audio':
            output_a = self.unimodal_audio_encoder(audio_input, audio_key_padding_mask)  # output_a.shape torch.Size([1, 1212, 768])
            if self.norm is not None:
                output_a = self.norm(output_a)
            return output_a        

        elif self.modality == 'text':
            output_t = self.unimodal_text_encoder(text_input,  src_key_padding_mask=~text_mask.bool()) # output_t.shape torch.Size([10, 1, 768]) # key_padding_mask shape (N,S)
            output_t = output_t.permute(1,0,2) # change shape from (S,N,E) to (B,S,E)    
            if self.norm is not None:
                output_t = self.norm(output_t)
            return output_t
        
        elif self.modality == 'video+audio':
            shared_neck = self.bottleneck.expand(-1, B, -1) # torch.Size([12, 8, 768])        
            output_v = self.unimodal_vision_encoder(video_input) # output_v.shape torch.Size([1, 12544, 768])
            output_a = self.unimodal_audio_encoder(audio_input, audio_key_padding_mask)  # output_a.shape torch.Size([1, 1212, 768])
            output_v = output_v.permute(1,0,2)
            output_a = output_a.permute(1,0,2)
            for col_v, col_a, pro_v, pro_a in  zip (self.token_collection_vision_layers, 
                                                    self.token_collection_audio_layers,
                                                    self.token_propagation_vision_layers,
                                                    self.token_propagation_audio_layers,):
                neck_vision = col_v(shared_neck, output_v)
                neck_audio = col_a(shared_neck, output_a, memory_key_padding_mask = audio_key_padding_mask)                
                shared_neck = (neck_vision + neck_audio) / 2.0
                output_v = pro_v(output_v, shared_neck)
                output_a = pro_a(output_a, shared_neck, tgt_key_padding_mask = audio_key_padding_mask)
            if self.norm is not None:
                output_v = self.norm(output_v)
                output_a = self.norm(output_a)
            output_v, output_a = output_v.permute(1,0,2), output_a.permute(1,0,2)
            return output_v, output_a

        elif self.modality == 'video+text':
            shared_neck = self.bottleneck.expand(-1, B, -1) # torch.Size([12, 8, 768])        
            output_v = self.unimodal_vision_encoder(video_input) # output_v.shape torch.Size([1, 12544, 768])
            output_t = self.unimodal_text_encoder(text_input,  src_key_padding_mask=~text_mask.bool()) # output_t.shape torch.Size([10, 1, 768]) # key_padding_mask shape (N,S)
            output_v = output_v.permute(1,0,2)
            for col_v, col_t, pro_v, pro_t in zip(self.token_collection_vision_layers,
                                                  self.token_collection_text_layers,
                                                  self.token_propagation_vision_layers,
                                                  self.token_propagation_text_layers):
                neck_vision = col_v(shared_neck, output_v)
                neck_text = col_t(shared_neck, output_t, memory_key_padding_mask = ~text_mask.bool() )
                shared_neck = (neck_vision + neck_text) / 2.0
                output_v = pro_v(output_v, shared_neck)
                output_t = pro_t(output_t, shared_neck, tgt_key_padding_mask =  ~text_mask.bool())
            if self.norm is not None:
                output_v = self.norm(output_v)
                output_t = self.norm(output_t)
            output_v, output_t = output_v.permute(1,0,2), output_t.permute(1,0,2)
            return output_v, output_t
        elif self.modality == 'audio+text':
            # pass 
            shared_neck = self.bottleneck.expand(-1, B, -1) # torch.Size([12, 8, 768])        
            output_a = self.unimodal_audio_encoder(audio_input, audio_key_padding_mask)  # output_a.shape torch.Size([1, 1212, 768])
            output_t = self.unimodal_text_encoder(text_input,  src_key_padding_mask=~text_mask.bool()) # output_t.shape torch.Size([10, 1, 768]) # key_padding_mask shape (N,S)
            output_a = output_a.permute(1,0,2)
            # shared encoders
            for col_a, col_t, pro_a, pro_t in zip(self.token_collection_audio_layers,
                                                  self.token_collection_text_layers,
                                                  self.token_propagation_audio_layers,
                                                  self.token_propagation_text_layers):
                neck_audio = col_a(shared_neck, output_a, memory_key_padding_mask = audio_key_padding_mask)
                neck_text = col_t(shared_neck, output_t, memory_key_padding_mask = ~text_mask.bool())
                shared_neck = (neck_audio + neck_text) / 2.0
                output_a = pro_a(output_a, shared_neck, tgt_key_padding_mask = audio_key_padding_mask)
                output_t = pro_t(output_t, shared_neck, tgt_key_padding_mask = ~text_mask.bool())
            if self.norm is not None:
                output_a = self.norm(output_a)
                output_t = self.norm(output_t)
            output_a, output_t = output_a.permute(1,0,2), output_t.permute(1,0,2)
            return output_a, output_t
        else:
            raise NotImplementedError


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

    def forward(self, x):
        return self.linear1(x)*self.linear2(x).sigmoid()