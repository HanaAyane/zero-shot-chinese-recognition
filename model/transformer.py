from typing import Optional
import torch.nn as nn
from torch.functional import Tensor
from torch.nn.modules.linear import Linear
import torch.nn.functional as F
import numpy as np
import torch


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q,
                              k,
                              value=src,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer(
            'pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return self.pos_table[:, :x.size(1)].clone().detach()


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                encoder_layer(d_model=1024, nhead=1, dim_feedforward=1024))
        self.layers = nn.Sequential(*layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        for layer in self.layers:
            output = layer(output,
                           src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask,
                           pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 nhead=4,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 layer_norm_eps=1e-5):
        super(TransformerDecoderLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)

        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout)
        # 兼容新版 PyTorch（2.0+）：nn.TransformerDecoder 内部会访问 self_attn 属性
        self.self_attn = self.multihead_attn

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = nn.ReLU(inplace=True)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        if isinstance(tgt, tuple):
            tgt = tgt[0]
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2, attn_map = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn_map


class TransformerDecoder(nn.Module):
    """自定义 TransformerDecoder，避免新版 PyTorch nn.TransformerDecoder 的兼容性问题。"""

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [decoder_layer] +
            [TransformerDecoderLayer(
                d_model=decoder_layer.norm1.normalized_shape[0],
                nhead=decoder_layer.multihead_attn.num_heads,
                dim_feedforward=decoder_layer.linear1.out_features,
                dropout=decoder_layer.dropout.p,
            ) for _ in range(num_layers - 1)]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory):
        output = tgt
        attn_map = None
        for layer in self.layers:
            output, attn_map = layer(output, memory)
        if self.norm is not None:
            output = self.norm(output)
        return output, attn_map


class ZSDecoder(nn.Module):

    def __init__(self,
                 decoder_embedding=1024,
                 dim_feedforward=2048,
                 n_head = 4,
                 num_layers_decoder=1):
        super(ZSDecoder, self).__init__()
        decoder_dropout = 0.1
        layer_decode = TransformerDecoderLayer(d_model=decoder_embedding,
                                               nhead=n_head,
                                               dim_feedforward=dim_feedforward,
                                               dropout=decoder_dropout)
        self.decoder = TransformerDecoder(layer_decode,
                                          num_layers=num_layers_decoder)

        self.duplicate_pooling = torch.nn.Parameter(
            torch.Tensor(decoder_embedding, 1))
        self.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(1))

        torch.nn.init.xavier_normal_(self.duplicate_pooling)
        torch.nn.init.constant_(self.duplicate_pooling_bias, 0)

    def forward(self, x, query_embed):

        if len(x.shape) == 4:
            embedding_spatial = x.flatten(2).transpose(1, 2)
        else:
            embedding_spatial = x
        bs = embedding_spatial.shape[0]
        tgt = query_embed.unsqueeze(1).expand(-1, bs, -1)
        h, attn_map = self.decoder(tgt, embedding_spatial.transpose(0, 1))
        h = h.transpose(0, 1)  # [bs, class, dim]
        out_extrap = torch.matmul(h, self.duplicate_pooling)
        h_out = out_extrap.flatten(1)
        h_out += self.duplicate_pooling_bias
        logits = h_out
        return logits, attn_map