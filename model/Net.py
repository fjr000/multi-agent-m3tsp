import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, embedding_dim, num_heads, use_FFN=True, hidden_size=128):
        super(CrossAttentionLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.use_FFN = use_FFN
        self.multiHeadAttention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True, )
        self.normalization1 = nn.LayerNorm(embedding_dim)
        if self.use_FFN:
            self.feedForward = nn.Sequential(nn.Linear(embedding_dim, hidden_size),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(hidden_size, embedding_dim))
            self.normalization2 = nn.LayerNorm(embedding_dim)

    def forward(self, q, k, v, attn_mask=None):
        assert q.shape[-1] == self.embedding_dim and k.shape[-1] == self.embedding_dim and len(q.shape) == len(
            k.shape) and q.shape[0] == k.shape[0] and k.shape == v.shape, \
            f"q.shape == k.shape == v.shape  is [B,S,{self.embedding_dim}]"

        att_out, _ = self.multiHeadAttention(q, k, v, attn_mask=attn_mask)
        hidden = att_out + q
        hidden_ln = self.normalization1(hidden)
        if self.use_FFN:
            fn_out = self.feedForward(hidden_ln)
            out = fn_out + hidden_ln
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
            assert len(xx.shape) == 3, "the shape of input data element must be [B, Seq, input_dim ]"
            assert xx.shape[-1] == dim, "the shape of input data element must be [B, Seq, input_dim ]"
            assert xx.shape[0] == x[0].shape[0], "the batch size of input data must be same"
        embeddings = []
        for i in range(self.embedding_layer_num):
            embedding = self.embedding_layers[i](x[i])
            embeddings.append(embedding)
        if len(x) > 1:
            embed = torch.cat(embeddings, dim=1)
        else:
            embed = embeddings[0]
        out = self.att(embed, attn_mask)
        split_dim = [dd.shape[1] for dd in x]
        out = torch.split(out, split_dim, dim=1)
        return out


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
