import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim
        self.key_dim = self.value_dim
        self.tanh_clipping = 10
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
                :param q: queries (batch_size, n_query, input_dim)
                :param h: data (batch_size, graph_size, input_dim)
                :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1), bool : True:mask False: allow
                Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
                :return:
                """
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)  # n_query = target_size in tsp

        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim

        h_flat = h.reshape(-1, input_dim)  # (batch_size*graph_size)*input_dim
        q_flat = q.reshape(-1, input_dim)  # (batch_size*n_query)*input_dim

        shape_k = (batch_size, target_size, -1)
        shape_q = (batch_size, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(shape_k)  # batch_size*targets_size*key_dim

        U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))  # batch_size*n_query*targets_size
        U = self.tanh_clipping * torch.tanh(U)

        if mask is not None:
            # mask = mask.view(batch_size, 1, target_size).expand_as(U)  # copy for n_heads times
            # U = U-1e8*mask  # ??
            U[mask] = -torch.inf
        # attention = torch.log_softmax(U, dim=-1)  # batch_size*n_query*targets_size

        # out = attention

        return U

class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module
        self.alpha = nn.Parameter(torch.Tensor([0]))

    def forward(self, input):
        return input + self.module(input) * self.alpha


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        self.att = nn.MultiheadAttention(
            embed_dim,
            n_heads,
            batch_first=True,
        )

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        if mask is not None:
            expand_mask = mask.unsqueeze(1).expand(-1,self.n_heads,-1,-1)
            out, _ = self.att(q, h, h, attn_mask = expand_mask)
        else:
            out, _ = self.att(q, h, h)

        return out


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d,
            'layer': nn.LayerNorm,
        }.get(normalization, None)

        if normalization == 'layer':
            self.normalizer = nn.LayerNorm(embed_dim)
        else:
            self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        elif isinstance(self.normalizer, nn.LayerNorm):
            return self.normalizer(input)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim=256, n_heads=8, norm=None):
        super(MultiHeadAttentionLayer, self).__init__()

        self.attn = MultiHeadAttention(n_heads, input_dim, embed_dim)

        if norm is None:
            self.attn_norm = None
            self.mlp_norm = None
        else:
            self.attn_norm = Normalization(embed_dim, normalization=norm)
            self.mlp_norm = Normalization(embed_dim, normalization=norm)

        self.alpha1 = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
        self.alpha2 = nn.Parameter(torch.zeros((1,), dtype=torch.float32))

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, q, kv=None, mask=None):
        o = self.attn(q, kv, mask)
        o = q + self.alpha1 * o
        n = o if self.attn_norm is None else self.attn_norm(o)
        o = self.mlp(n)
        o = n + self.alpha2 * o
        n = o if self.mlp_norm is None else self.mlp_norm(o)
        return n



