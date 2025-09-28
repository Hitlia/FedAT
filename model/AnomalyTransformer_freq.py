import torch
import torch.nn as nn
import torch.nn.functional as F

# from .attn_freq import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding
from .layers import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
from .autoCorrelation import AutoCorrelationLayer
from model.FourierCorrelation import FourierBlock, FourierCrossAttention
from model.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform


class AnomalyTransformer_freq(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=2, d_layers=1,d_ff=512,
                 dropout=0.0, activation='gelu', version='Wavelets', output_attention=True):
        super(AnomalyTransformer_freq, self).__init__()
        self.output_attention = output_attention
        self.seq_len = win_size
        
        kernel_size = [12,24]
        # if isinstance(kernel_size, list):
        #     self.decomp = series_decomp_multi(kernel_size)
        # else:
        #     self.decomp = series_decomp(kernel_size)

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(enc_in, d_model, dropout)
        
        L = 1
        base = 'legendre'
        modes = 32
        cross_activation = 'tanh'
        mode_select = 'random'
        
        if version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(win_size,ich=d_model, L=L, base=base,attention_dropout=dropout,output_attention=True)
            decoder_self_att = MultiWaveletTransform(win_size,ich=d_model, L=L, base=base,attention_dropout=dropout,output_attention=False)
            decoder_cross_att = MultiWaveletCross(in_channels=d_model,
                                                  out_channels=d_model,
                                                  seq_len_q=self.seq_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=modes,
                                                  ich=d_model,
                                                  base=base,
                                                  activation=cross_activation)
        else:
            encoder_self_att = FourierBlock(win_size,in_channels=d_model,
                                            out_channels=d_model,
                                            seq_len=self.seq_len,
                                            modes=modes,
                                            attention_dropout=dropout,
                                            mode_select_method=mode_select,
                                            output_attention=True)
            decoder_self_att = FourierBlock(win_size,in_channels=d_model,
                                            out_channels=d_model,
                                            seq_len=self.seq_len,
                                            modes=modes,
                                            attention_dropout=dropout,
                                            mode_select_method=mode_select,
                                            output_attention=False)
            decoder_cross_att = FourierCrossAttention(in_channels=d_model,
                                                      out_channels=d_model,
                                                      seq_len_q=self.seq_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=modes,
                                                      mode_select_method=mode_select)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        d_model, n_heads,output_attention=True),

                    d_model,
                    d_ff,
                    moving_avg=kernel_size,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        d_model, n_heads,output_attention=False),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        d_model, n_heads),
                    d_model,
                    c_out,
                    d_ff,
                    moving_avg=kernel_size,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

    def forward(self, x):
        # decomp init
        B, L, C = x.shape
        mean = torch.mean(x, dim=1).unsqueeze(1).repeat(1, L, 1)
        # seasonal_init, trend_init = self.decomp(x_enc)
        # # decoder input
        trend_init = mean
        seasonal_init = torch.zeros_like(trend_init)
        seasonal_init = x-trend_init
        # trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        # seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # enc
        enc_out = self.enc_embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out, attn_mask=None)
        # dec
        dec_out = self.dec_embedding(seasonal_init)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out, series, prior, sigmas
        else:
            return dec_out  # [B, L, D]