import torch
import torch.nn as nn
from model.RefModel.MHA import MultiHeadAttentionLayer, SingleHeadAttention


class AgentCityAttentionDecoder(nn.Module):
    def __init__(self, hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2, norm = "layer"):
        super(AgentCityAttentionDecoder, self).__init__()

        self.agent_city_att = nn.ModuleList(
            [
                MultiHeadAttentionLayer(embed_dim, embed_dim, hidden_dim, n_heads=num_heads, norm=norm)
                for _ in range(num_layers)
            ]
        )

        self._logp = SingleHeadAttention(embed_dim)

    def forward(self,agent_embed, city_embed, masks = None):
        """
        :param agent: [B,N,2]
        :return:
        """
        extra_masks = ~torch.eye(agent_embed.size(1), device=agent_embed.device).bool()[None,:].expand(agent_embed.size(0),-1,-1)
        expand_masks = torch.cat([masks, extra_masks], dim=-1)
        for model in self.agent_city_att:
            agent_embed = model(agent_embed, city_embed,mask = expand_masks)

        logits = self._logp(agent_embed, city_embed[:,:-agent_embed.size(1),:], masks)

        return agent_embed, logits

if __name__ == '__main__':
    from CityAttentionEncoder import CityAttentionEncoder
    from AgentAttentionEncoder import AgentAttentionEncoder
    encoder = CityAttentionEncoder(4, 128, 2, 2, 'batch', 256)
    city = torch.randn((2, 10, 2))
    city_mask = torch.randint(0, 2, (2, 10, 10)).bool()
    city_embed, city_embed_mean = encoder(city, city_mask)

    agent_encoder =AgentAttentionEncoder(input_dim=2, hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2, norm='batch')
    agent1 = torch.randint(0,10,(2,3,2))
    agent2 = torch.randn((2,3,8))
    agents_mask = torch.randint(0, 2, (2, 3, 3)).bool()
    agents_embed = agent_encoder(city_embed, city_embed_mean, torch.cat([agent1, agent2], dim=-1), agents_mask)

    agent_city_decoer = AgentCityAttentionDecoder(hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2, norm = "batch")
    agent_city_mask = torch.randint(0, 2, (2, 3, 10)).bool()
    o = agent_city_decoer(agents_embed, city_embed, agent_city_mask)