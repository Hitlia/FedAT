import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange, reduce, repeat

from .attn_new import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding, DataEmbedding_ch


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        
        self.conv1_time = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2_time = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1_time = nn.LayerNorm(d_model)
        self.norm2_time = nn.LayerNorm(d_model)
        self.dropout_time = nn.Dropout(dropout)
        
        self.conv1_channel = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2_channel = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1_channel = nn.LayerNorm(d_model)
        self.norm2_channel = nn.LayerNorm(d_model)
        self.dropout_channel = nn.Dropout(dropout)
        
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x_time, x_channel, attn_mask=None):
        new_x_time, new_x_channel, attn, mask = self.attention(
            x_time, x_time, x_time, x_channel, x_channel, x_channel,
            attn_mask=attn_mask
        )
        
        x_time = x_time + self.dropout_time(new_x_time)
        y_time = x_time = self.norm1_time(x_time)
        y_time = self.dropout_time(self.activation(self.conv1_time(y_time.transpose(-1, 1))))
        y_time = self.dropout_time(self.conv2_time(y_time).transpose(-1, 1))
        
        x_channel = x_channel + self.dropout_channel(new_x_channel)
        y_channel = x_channel = self.norm1_channel(x_channel)
        y_channel = self.dropout_channel(self.activation(self.conv1_channel(y_channel.transpose(-1, 1))))
        y_channel = self.dropout_channel(self.conv2_channel(y_channel).transpose(-1, 1))

        return self.norm2_time(x_time + y_time), self.norm2_channel(x_channel + y_channel), attn, mask


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm_time = norm_layer
        self.norm_channel = norm_layer

    def forward(self, x_time, x_channel, attn_mask=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        for attn_layer in self.attn_layers:
            x_time, x_channel, series, prior = attn_layer(x_time, x_channel, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)

        if self.norm_time is not None:
            x_time = self.norm_time(x_time)
        
        if self.norm_channel is not None:
            x_channel = self.norm_channel(x_channel)

        return x_time, x_channel, series_list, prior_list


class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, num_points, enc_in, c_out, num_points_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True):
        super(AnomalyTransformer, self).__init__()
        self.channels_num = enc_in
        self.num_points = num_points
        self.output_attention = output_attention
        
        # self.encoder1 = nn.Linear(1,enc_in)
        # Encoding
        self.embedding_time = DataEmbedding(enc_in, d_model, dropout)
        self.embedding_channel = DataEmbedding_ch(num_points, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, enc_in,n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection_time = nn.Linear(d_model, c_out, bias=True)
        self.projection_channel = nn.Linear(d_model, num_points_out, bias=True)
        
        self.proj_time = nn.Linear(num_points, 1, bias=True)
        self.proj_channel = nn.Linear(enc_in, 1, bias=True)

    def forward(self, x):
        # print(x.shape)
        # x = self.encoder1(x)
        # print(x.shape)
        enc_out_time = self.embedding_time(x)
        enc_out_channel = self.embedding_channel(x.permute(0,2,1))
        # print(x.shape)
        # print(enc_out.shape)
        enc_out_time, enc_out_channel, series, prior = self.encoder(enc_out_time, enc_out_channel)
        # print(series[0].shape)
        # print(prior[0].shape)
        
        enc_out_time = self.projection_time(enc_out_time)
        enc_out_channel = self.projection_channel(enc_out_channel)
        enc_out = enc_out_time + enc_out_channel.permute(0,2,1)
        # print(enc_out.shape)
        # print(len(series))
        # print(series[0])
        # print(prior.shape)
        series_list = []
        prior_list = []
        for i in range(len(series)):
            series_1 = torch.softmax(self.proj_time(series[i]).squeeze(), dim=-1)
            prior_1 = torch.softmax(self.proj_channel(prior[i]).squeeze(), dim=-1)
            # print(series_1)
            # print(prior_1.shape)
            series_1 = series_1.repeat(self.channels_num,1,1,1).permute(1,2,3,0)
            prior_1 = prior_1.repeat(self.num_points,1,1,1).permute(1,2,0,3)
            # print(series_1.shape)
            # print(prior_1.shape)
            series_list.append(series_1)
            prior_list.append(prior_1)

        if self.output_attention:
            return enc_out, series_list, prior_list
        else:
            return enc_out  # [B, L, D]
