import torch
import torch.nn as nn
from model.RefModel.MHA import MultiHeadAttentionLayer


class CityAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            input_dim=None,
            norm=None,
            hidden_dim=512
    ):
        super(CityAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(input_dim, embed_dim) if input_dim is not None else None

        self.attn_layers = nn.ModuleList(
            [
                MultiHeadAttentionLayer(embed_dim, embed_dim, hidden_dim, n_heads, norm)
                for _ in range(n_layers)
            ]
        )

    def forward(self, city, mask=None):
        # Batch multiply to get initial embeddings of nodes

        h = self.init_embed(city) if self.init_embed is not None else city

        for attn in self.attn_layers:
            h = attn(h, mask=mask)

        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(keepdim = True, dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )


if __name__ == "__main__":
    encoder = CityAttentionEncoder(4, 128, 2, 2, 'batch', 256)
    city = torch.randn((2, 10, 2))
    city_mask = torch.randint(0, 2, (2, 10, 10)).bool()
    o = encoder(city, city_mask)
