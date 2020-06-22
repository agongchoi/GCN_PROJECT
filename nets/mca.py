# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from nets.net_utils import FC, MLP, LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch, math


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, args):
        super(MHAtt, self).__init__()
        self.num_partitionings = args.num_partitionings
        #self.feature_dim = feature_dim
        self.hidden_dim = args.hidden_dim
        self.multi_head = args.multi_head
        self.dropout_rate = 0.1
        self.hidden_size_head = int(self.hidden_dim / self.multi_head)

        self.linear_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_k = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_q = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.linear_merge = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.multi_head,
            self.hidden_size_head
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.multi_head,
            self.hidden_size_head
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.multi_head,
            self.hidden_size_head
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hidden_dim
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

class FFN(nn.Module):
    def __init__(self, args):
        super(FFN, self).__init__()

        self.dropout_rate = 0.1
        self.ff_size = args.hidden_dim * 4

        self.mlp = MLP(
            in_size=args.hidden_dim,
            mid_size=self.ff_size,
            out_size=args.hidden_dim,
            dropout_r=self.dropout_rate,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)

class SA(nn.Module):
    def __init__(self, args):
        super(SA, self).__init__()

        self.dropout_rate = 0.1

        self.mhatt = MHAtt(args)
        self.ffn = FFN(args)

        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.norm1 = LayerNorm(args.hidden_dim)

        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.norm2 = LayerNorm(args.hidden_dim)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# ------------------------------------------------
# ---- MSA Layers stacking ----
# ------------------------------------------------

class MSA_ST(nn.Module):
    def __init__(self,args):
        super(MSA_ST, self).__init__()

        self.enc_list = nn.ModuleList([SA(args) for _ in range(args.sa_layer)])

    def forward(self, x, x_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)

        return x



