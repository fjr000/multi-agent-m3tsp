from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.w = nn.Linear(dim, hidden_dim * 2)
        self.out = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        a, b = self.w(x).chunk(2, dim=-1)
        h = F.silu(a) * b
        out = self.out(h)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.layer = nn.MultiheadAttention(embedding_dim, n_heads, batch_first=True)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        out, _ = self.layer(x, x, x,
                            key_padding_mask=key_padding_mask,
                            attn_mask=attn_mask)
        return out


class CrossAttentionLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_size=128):
        super(CrossAttentionLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.multiHeadAttention = nn.MultiheadAttention(embedding_dim, num_heads,
                                                        batch_first=True,
                                                        )
        self.normalization1 = nn.LayerNorm(embedding_dim)
        self.alpha1 = nn.Parameter(torch.tensor([0.1], requires_grad=True))
        self.alpha2 = nn.Parameter(torch.tensor([0.1], requires_grad=True))

        self.ff = SwiGLU(embedding_dim, hidden_size)

        self.normalization2 = nn.LayerNorm(embedding_dim)

    def forward(self, q, k, v, attn_mask=None):
        q_norm = self.normalization1(q)
        att_out, _ = self.multiHeadAttention(q_norm, k, v, attn_mask = attn_mask)
        hidden = q + self.alpha1 * att_out

        hidden_norm = self.normalization2(hidden)
        ff_out = self.ff(hidden_norm)
        out = hidden + self.alpha2 * ff_out
        return out


class CrossAttentionCacheKVLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_size=128):
        super(CrossAttentionCacheKVLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.multiHeadAttention = MultiHeadAttentionCacheKV(embedding_dim, num_heads)
        self.normalization1 = nn.LayerNorm(embedding_dim)
        self.normalization2 = nn.LayerNorm(embedding_dim)
        self.alpha1 = nn.Parameter(torch.tensor([0.1], requires_grad=True))
        self.alpha2 = nn.Parameter(torch.tensor([0.1], requires_grad=True))

        self.ff = SwiGLU(embedding_dim, hidden_size)

    def cache_keys(self, city_embed):
        self.multiHeadAttention.cache_keys(city_embed)

    def forward(self, q, attn_mask=None, batch_mask=None):
        q_norm = self.normalization1(q)
        att_out = self.multiHeadAttention(q_norm, attn_mask, batch_mask)
        hidden = q + self.alpha1 * att_out

        hidden_norm = self.normalization2(hidden)
        ff_out = self.ff(hidden_norm)
        out = hidden + self.alpha2 * ff_out
        return out


class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module
        self.alpha = nn.Parameter(torch.tensor([0.1], requires_grad=True))

    def forward(self, inputs, key_padding_mask=None, attn_mask=None):
        if key_padding_mask is not None or attn_mask is not None:
            skip = self.alpha * self.module(inputs, key_padding_mask, attn_mask)
        else:
            skip = self.alpha * self.module(inputs)
        return inputs + skip


class MultiHeadAttentionCacheKV(nn.Module):
    def __init__(self, d_model, n_head, dropout=0):
        super(MultiHeadAttentionCacheKV, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        assert d_model % n_head == 0
        self.head_dim = d_model // n_head

        self.W_kv = nn.Linear(self.d_model, self.d_model * 2, bias=False)
        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.glimpse_K = None
        self.glimpse_V = None

    def cache_keys(self, embed):
        if self.W_kv.weight.device != embed.device:
            self.W_kv = self.W_kv.to(embed.device)
        B, A, E = embed.shape
        self.glimpse_K, self.glimpse_V = self.W_kv(embed).chunk(2, dim=-1)
        self.glimpse_K = self.glimpse_K.view(B, -1, self.n_head, self.head_dim).permute(0, 2, 3, 1)
        self.glimpse_V = self.glimpse_V.view(B, -1, self.n_head, self.head_dim).transpose(1, 2)

    def forward(self, q_embed, mask=None, batch_mask = None):
        B = q_embed.size(0)
        glimpse_Q = self.W_q(q_embed).view(B, -1, self.n_head, self.head_dim).transpose(1, 2)

        glimpse_K = self.glimpse_K
        glimpse_V = self.glimpse_V
        if batch_mask is not None:
            glimpse_K = glimpse_K[batch_mask]
            glimpse_V = glimpse_V[batch_mask]

        score = torch.matmul(glimpse_Q, glimpse_K) / math.sqrt(self.head_dim)

        if mask is not None:
            score = score.masked_fill(mask.unsqueeze(1), -torch.inf)

        attn_weight = F.softmax(score, dim=-1)
        attn_weight = self.dropout(attn_weight)

        context = torch.matmul(attn_weight, glimpse_V)
        context = context.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        out = self.out(context)
        return out

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model: int, tanh_clip: float = 10.0, use_query_proj: bool = False, use_cache: bool = False, ):
        super().__init__()
        self.d_model = d_model
        self.tanh_clip = tanh_clip
        self.use_query_proj = use_query_proj
        self.norm_factor = 1 / math.sqrt(d_model)
        self.use_cache = use_cache

        # Key projection layer
        self.W_k = nn.Linear(d_model, d_model, bias=False)

        # Optional Query projection
        if use_query_proj:
            self.W_q = nn.Linear(d_model, d_model, bias=False)
        else:
            self.register_module('W_q', None)

        # 缓存的 key（shape: [B, d_model, N]）
        self.register_buffer('glimpse_K', None)

    def cache_keys(self, embed: torch.Tensor) -> None:
        """
        缓存输入 embedding 的 Key 向量。
        Args:
            embed: [B, N, d_model]
        """
        # 确保 embed 是连续内存
        if not embed.is_contiguous():
            embed = embed.contiguous()
        # 计算并缓存 K
        self.glimpse_K = self.W_k(embed).transpose(-2, -1)  # shape: [B, d_model, N]

    def forward(
            self,
            q_embed: torch.Tensor,  # shape: [B, H, d_model]
            k_embed: torch.Tensor = None,  # shape: [B, N, d_model]
            mask: Optional[torch.BoolTensor] = None,  # shape: [B, H, N]
            batch_mask: Optional[torch.BoolTensor] = None  # shape: [B]
    ) -> torch.Tensor:
        """
        Args:
            q_embed: Query embeddings, shape [B, H, d_model]
            mask: Mask for invalid positions, shape [B, H, N]
            batch_mask: Boolean mask to select batch elements, shape [B]

        Returns:
            logits: Attention logits, shape [B, H, N]
        """
        B, H, _ = q_embed.shape

        # Apply query projection if enabled
        if self.use_query_proj:
            glimpse_Q = self.W_q(q_embed)  # shape: [B, H, d_model]
        else:
            glimpse_Q = q_embed

        if not self.use_cache:
            # 不使用缓存，则每次都计算
            self.cache_keys(k_embed)
        glimpse_K = self.glimpse_K  # shape: [B, d_model, N]

        # 如果有 batch_mask，则筛选对应的 batch
        if batch_mask is not None:
            glimpse_K = glimpse_K[batch_mask]  # shape: [B', d_model, N]

        # 点积计算 logits = Q @ K / sqrt(d_model)
        logits = torch.bmm(glimpse_Q, glimpse_K) * self.norm_factor

        # 应用 tanh clipping
        if self.tanh_clip > 0:
            logits = torch.tanh(logits) * self.tanh_clip

        # 应用 mask
        if mask is not None:
            assert mask.dtype == torch.bool
            logits = logits.masked_fill(mask, -torch.inf)

        return logits


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
