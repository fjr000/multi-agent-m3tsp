import torch
import torch.nn as nn
import torch.nn.functional as F
import model.Base.Net as Net


class Encoder(nn.Module):
    def __init__(self, input_dim, n_heads, embedding_dim, n_layers, hidden_dim, normalization='batch'):
        super(Encoder, self).__init__()
        self.init_embed = nn.Linear(input_dim, embedding_dim)

        self.encoder = nn.Sequential(
            *[
                Net.MultiHeadAttentionLayer(n_heads, embedding_dim, hidden_dim, normalization)
                for _ in range(n_layers)
            ]
        )

    def forward(self, input):
        h = self.init_embed(input)

        h = self.encoder(h)

        return (
            h,
            h.mean(dim=1),
        )


if __name__ == '__main__':
    city_dim = 2
    GE = Encoder(2, 8, 128, 2, 256, normalization='batch')
    from envs.GraphGenerator import GraphGenerator
    from utils.TensorTools import _convert_tensor

    GG = GraphGenerator(2, 10, 2, None)
    graph = GG.generate()
    graph_tensor = _convert_tensor(graph)
    embeds, embeds_mean = GE(graph_tensor)
    print(embeds)
    print(embeds_mean)
