import torch
import torch.nn as nn
from model.RefModel.MHA import MultiHeadAttentionLayer
from model.RefModel.PositionEncoder import PositionalEncoder

class CityAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            input_dim=None,
            norm='batch',
            hidden_dim=512
    ):
        super(CityAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_depot_embed = nn.Linear(input_dim, embed_dim) if input_dim is not None else None
        self.init_city_embed = nn.Linear(input_dim, embed_dim) if input_dim is not None else None

        self.attn_layers = nn.ModuleList(
            [
                MultiHeadAttentionLayer(embed_dim, embed_dim, hidden_dim, n_heads, norm)
                for _ in range(n_layers)
            ]
        )

        self.position_encoder = PositionalEncoder(embed_dim)
        self.pos_embed_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, city, n_agents, mask=None):
        # Batch multiply to get initial embeddings of nodes
        depot_embed = self.init_depot_embed(city[:,0:1,:])
        city_embed = self.init_city_embed(city[:,1:,:]) if self.init_city_embed is not None else city

        pos_embed = self.position_encoder(n_agents+1)
        depot_embed_repeat = depot_embed.expand(-1, n_agents+1, -1)
        pos_embed = self.alpha * self.pos_embed_proj(pos_embed) / n_agents
        depot_pos_embed = depot_embed_repeat + pos_embed[None,:]

        graph_embed = torch.cat([depot_pos_embed[:,0:1,:], city_embed, depot_pos_embed[:,1:,:]], dim=1)
        for attn in self.attn_layers:
            graph_embed = attn(graph_embed, mask=mask)

        return (
            graph_embed,  # (B,A+N,E)
            graph_embed.mean(keepdim = True, dim=1) # (batch_size, 1, embed_dim) mean(A+E)
        )


if __name__ == "__main__":
    encoder = CityAttentionEncoder(4, 128, 2, 2, 'batch', 256)
    city = torch.randn((2, 10, 2))
    city_mask = torch.randint(0, 2, (2, 10, 10)).bool()
    o = encoder(city, city_mask)
