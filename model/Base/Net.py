import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.GRU):
            for param in layer.parameters():
                if param.dim() > 1:  # 仅对 weight 参数初始化
                    nn.init.orthogonal_(param)

def make_fc_layer(in_features: int, out_features: int, use_bias=True, layer_norm=False):
    """Wrapper function to create and initialize a linear layer
    Args:
        in_features (int): ``in_features``
        out_features (int): ``out_features``

    Returns:
        nn.Linear: the initialized linear layer
    """

    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)

    nn.init.orthogonal(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)
    if layer_norm:
        seq = nn.Sequential(
            nn.LayerNorm(in_features),
            fc_layer
        )
        return seq
    return fc_layer


class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module
        self.alpha = nn.Parameter(torch.tensor([0.1]))

    def forward(self, inputs, key_padding_mask = None, attn_mask = None):
        if key_padding_mask is not None or attn_mask is not None:
            return inputs + self.alpha * self.module(inputs, key_padding_mask, attn_mask)
        else:
            return inputs + self.alpha * self.module(inputs)


class MLP(nn.Module):
    def __init__(self, layers_dim, nonlinear_layer_last=True):
        super(MLP, self).__init__()
        self.seq = nn.Sequential()
        for i, layer in enumerate(layers_dim):
            if i < len(layers_dim) - 1:
                self.seq.append(make_fc_layer(layer, layers_dim[i + 1]))
                if i < len(layers_dim) - 2 or (nonlinear_layer_last and i == len(layers_dim) - 2):
                    self.seq.append(nn.ReLU())

    def forward(self, x):
        return self.seq(x)


class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(EmbeddingNet, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.embedding = MLP([input_dim, 256, embedding_dim], nonlinear_layer_last=False)

    def forward(self, x):
        return self.embedding(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads, dropout = 0):
        super(MultiHeadAttention, self).__init__()
        self.layer = nn.MultiheadAttention(embedding_dim, n_heads, dropout = dropout, batch_first = True)

    def forward(self, x, key_padding_mask = None,attn_mask = None):
        out,_ = self.layer(x, x, x, key_padding_mask = key_padding_mask,attn_mask = attn_mask)
        return out

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_heads, embedding_dim, hidden_dim, normalization='batch', dropout = 0):
        super(MultiHeadAttentionLayer, self).__init__()

        self.attention = SkipConnection(
                    MultiHeadAttention(
                        embedding_dim,
                        n_heads,
                        dropout=dropout
                    )
                )
        # self.alpha1 = nn.Parameter(torch.tensor([0.1]))
        self.norm1 = nn.BatchNorm1d(embedding_dim, affine=True) if normalization == 'batch' else nn.LayerNorm(embedding_dim)

        self.mlp = SkipConnection(
                    nn.Sequential(
                        nn.Linear(embedding_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, embedding_dim)
                    )
                )
        # self.alpha2 = nn.Parameter(torch.tensor([0.1]))
        self.norm2 = nn.BatchNorm1d(embedding_dim, affine=True) if normalization == 'batch' else nn.LayerNorm(embedding_dim)
        self.embed_dim = embedding_dim

    def forward(self, x, key_padding_mask = None,masks = None):
        o = self.attention(x, key_padding_mask = key_padding_mask, attn_mask = masks)
        if isinstance(self.norm1, nn.BatchNorm1d):
            _shape = o.shape
            x = self.norm1(o.reshape(-1, self.embed_dim))
            o = self.mlp(x, None)
            o = self.norm2(o).reshape(*_shape)
        elif isinstance(self.norm2, nn.LayerNorm):
            x = self.norm1(o)
            o = self.mlp(x, None)
            o = self.norm2(o)
        else:
            raise NotImplementedError
        return o


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


class SelfAttentionLayer(nn.Module):
    # For not self attention
    def __init__(self, embedding_dim, num_heads, use_FFN=True, hidden_size=128):
        super(SelfAttentionLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.use_FFN = use_FFN
        self.multiHeadAttention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True, )
        self.normalization1 = nn.LayerNorm(embedding_dim)
        if self.use_FFN:
            self.feedForward = nn.Sequential(nn.Linear(embedding_dim, hidden_size),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(hidden_size, embedding_dim))
            self.normalization2 = nn.LayerNorm(embedding_dim)

    def forward(self, x, attn_mask=None):
        assert x.shape[-1] == self.embedding_dim, \
            f"q.shape == k.shape == v.shape  is [B,S,{self.embedding_dim}]"

        input_ln = self.normalization1(x)
        att_out, _ = self.multiHeadAttention(input_ln, input_ln, input_ln, attn_mask=attn_mask)
        hidden = att_out + x
        hidden_ln = self.normalization2(hidden)
        if self.use_FFN:
            fn_out = self.feedForward(hidden_ln)
            out = fn_out + hidden
        else:
            out = hidden
        return out


class CrossAttentionLayer(nn.Module):
    # For not self attention
    def __init__(self, embedding_dim, num_heads, use_FFN=True, hidden_size=128,dropout=0):
        super(CrossAttentionLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.use_FFN = use_FFN
        self.multiHeadAttention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True, dropout=dropout)
        self.normalization1 = nn.LayerNorm(embedding_dim)
        self.alpha1 = nn.Parameter(torch.zeros(1))
        self.alpha2 = nn.Parameter(torch.zeros(1))
        if self.use_FFN:
            self.feedForward = nn.Sequential(nn.Linear(embedding_dim, hidden_size),
                                             nn.GELU(),
                                             nn.Linear(hidden_size, embedding_dim))
            self.normalization2 = nn.LayerNorm(embedding_dim)

    def forward(self, q, k, v, attn_mask=None):
        assert q.shape[-1] == self.embedding_dim and k.shape[-1] == self.embedding_dim and len(q.shape) == len(
            k.shape) and q.shape[0] == k.shape[0] and k.shape == v.shape, \
            f"q.shape == k.shape == v.shape  is [B,S,{self.embedding_dim}]"

        att_out, _ = self.multiHeadAttention(q, k, v, attn_mask=attn_mask)
        hidden = self.alpha1 * att_out + q
        hidden_ln = self.normalization1(hidden)
        if self.use_FFN:
            fn_out = self.feedForward(hidden_ln)
            out = self.alpha2 * fn_out + hidden_ln
            out = self.normalization2(out)
        else:
            out = hidden_ln
        return out


class SelfEncoder(nn.Module):
    def __init__(self, input_dims, embedding_dim, num_heads, use_FFN=True, hidden_size=128):
        assert isinstance(input_dims, list) or isinstance(input_dims, tuple), "input_dims must be a list or tuple"
        super(SelfEncoder, self).__init__()
        self.embedding_layer_num = len(input_dims)
        self.embedding_layers = nn.ModuleList()
        self.embedding_dim = embedding_dim
        self.input_dims = input_dims
        for i in range(self.embedding_layer_num):
            self.embedding_layers.append(EmbeddingNet(input_dims[i], embedding_dim))

        self.att = SelfAttentionLayer(embedding_dim, num_heads, use_FFN, hidden_size)

    def forward(self, x, attn_mask=None):
        assert isinstance(x, list) or isinstance(x, tuple), "input must be a list or tuple"
        assert len(x) == self.embedding_layer_num, f"input shoule be split into {self.embedding_layer_num} type"
        for xx, dim in zip(x, self.input_dims):
            # assert len(xx.shape) == 3, "the shape of input data element must be [B, Seq, input_dim ]"
            assert xx.shape[-1] == dim, "the shape of input data element must be [B, Seq, input_dim ]"
            assert xx.shape[0] == x[0].shape[0], "the batch size of input data must be same"
        embeddings = []
        for i in range(self.embedding_layer_num):
            embedding = self.embedding_layers[i](x[i])
            embeddings.append(embedding)
        if len(x) > 1:
            embed = torch.cat(embeddings, dim=-2)
        else:
            embed = embeddings[0]
        out = self.att(embed, attn_mask)
        split_dim = [dd.shape[1] for dd in x]
        out = torch.split(out, split_dim, dim=-2)
        return out


class SingleDecoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, use_FFN=True, hidden_size=128):
        super(SingleDecoder, self).__init__()

        self.CA1 = CrossAttentionLayer(embedding_dim, num_heads, use_FFN, hidden_size)
        self.CA2 = CrossAttentionLayer(embedding_dim, num_heads, use_FFN, hidden_size)
        self.SAO = SingleHeadAttention(embedding_dim)

    def forward(self, agents_embeddings, cities_embeddings, attn_mask=None):
        h_c = self.CA1(agents_embeddings, cities_embeddings, cities_embeddings, attn_mask)
        h_c_embed = self.SAO(h_c, cities_embeddings, attn_mask)
        next_idx = torch.argmax(h_c_embed, dim=-2)

        return next_idx


if __name__ == '__main__':
    import envs.GraphGenerator as G

    GG = G.GraphGenerator()
    data = GG.generate()
    data = torch.tensor(data, dtype=torch.float32)
    mlp = MLP([2, 256, 128], nonlinear_layer_last=False)
    mlp_out = mlp(data)
    print(mlp_out)

    embedding = EmbeddingNet(2, 128)
    embedding_out = embedding(data)
    print(embedding_out)

    att = SelfAttentionLayer(128, 4, use_FFN=True)
    att_out = att(embedding_out, attn_mask=None)
    print(att_out)

    CA = CrossAttentionLayer(128, 4, use_FFN=True)
    CA_out = CA(embedding_out[:, 2, :].unsqueeze(1), embedding_out, embedding_out, attn_mask=None)
    print(CA_out)

    SE = SelfEncoder([2], 128, 4, use_FFN=True)
    SE_out = SE([data])
    print(SE_out)

    SE_2 = SelfEncoder([2, 2, 2], 128, 4, use_FFN=True)
    SE_2_out = SE_2([data[:, 0].unsqueeze(1), data[:, 1:3], data[:, 3:]])
    print(SE_2_out)
