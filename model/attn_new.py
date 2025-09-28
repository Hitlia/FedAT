import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size

    def forward(self, queries_time, keys_time, values_time, queries_channel, keys_channel, values_channel, attn_mask):
        B, L, H, E = queries_time.shape
        _, C, _, _ = queries_channel.shape
        _, S, _, D = values_time.shape
        _, N, _, D = values_channel.shape
        scale = self.scale or 1. / sqrt(E)

        scores_time = torch.einsum("blhe,bshe->bhls", queries_time, keys_time)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries_time.device)
            scores_time.masked_fill_(attn_mask.mask, -np.inf)
        attn_time = scale * scores_time

        series = self.dropout(torch.softmax(attn_time, dim=-1))
        V_time = torch.einsum("bhls,bshd->blhd", series, values_time)
        
        scores_channel = torch.einsum("bche,bnhe->bhcn", queries_channel, keys_channel)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, C, device=queries_channel.device)
            scores_channel.masked_fill_(attn_mask.mask, -np.inf)
        attn_channel = scale * scores_channel

        prior = self.dropout(torch.softmax(attn_channel, dim=-1))
        V_channel = torch.einsum("bhcn,bnhd->bchd", prior, values_channel)

        if self.output_attention:
            return (V_time.contiguous(), V_channel.contiguous(), series, prior)
        else:
            return (V_time.contiguous(), V_channel.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, num_channels, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection_time = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection_time = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection_time = nn.Linear(d_model,
                                          d_values * n_heads)
        self.query_projection_channel = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection_channel = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection_channel = nn.Linear(d_model,
                                          d_values * n_heads)
        self.out_projection_time = nn.Linear(d_values * n_heads, d_model)
        self.out_projection_channel = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries_time, keys_time, values_time, queries_channel, keys_channel, values_channel, attn_mask):
        B, L, _ = queries_time.shape
        B, C, _ = queries_channel.shape
        _, S, _ = keys_time.shape
        _, N, _ = keys_channel.shape
        H = self.n_heads
        # x = queries_time
        queries_time = self.query_projection_time(queries_time).view(B, L, H, -1)
        keys_time = self.key_projection_time(keys_time).view(B, S, H, -1)
        values_time = self.value_projection_time(values_time).view(B, S, H, -1)
        
        queries_channel = self.query_projection_channel(queries_channel).view(B, C, H, -1)
        keys_channel = self.key_projection_channel(keys_channel).view(B, N, H, -1)
        values_channel = self.value_projection_channel(values_channel).view(B, N, H, -1)

        out_time, out_channel, series, prior = self.inner_attention(
            queries_time,
            keys_time,
            values_time,
            queries_channel,
            keys_channel,
            values_channel,
            attn_mask
        )
        # print(out_time.shape)
        # print(out_channel.shape)
        out_time = out_time.view(B, L, -1)
        out_channel = out_channel.view(B, C, -1)

        return self.out_projection_time(out_time), self.out_projection_channel(out_channel), series, prior
