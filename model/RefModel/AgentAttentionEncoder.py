import torch
import torch.nn as nn
from model.RefModel.MHA import MultiHeadAttentionLayer
from model.RefModel.PositionEncoder import PositionalEncoder

class AgentEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(AgentEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.depot_pos_embed = nn.Linear(self.embed_dim , self.embed_dim)
        self.distance_cost_embed = nn.Linear(2, self.embed_dim)
        self.next_cost_embed = nn.Linear(3, self.embed_dim)
        self.problem_scale_embed = nn.Linear(1, self.embed_dim)
        self.graph_embed = nn.Linear(self.embed_dim, self.embed_dim)

        self.position_embed = PositionalEncoder(self.embed_dim)

        self.agent_embed = nn.Linear(2 * self.embed_dim, self.embed_dim)

    def forward(self,cities_embed, graph_embed, agent_state):
        """
        :param graph_embed
        :param agent_state: [B,M,14]
        :return:
        """

        # cities_expand = cities_embed.expand(agent_state.size(0), -1, -1)
        depot_pos = cities_embed[torch.arange(agent_state.size(0))[:, None, None], agent_state[:,:,:2].long(),:]
        depot_pos_embed = self.depot_pos_embed(depot_pos[:,:,0,:] - depot_pos[:,:,1,:])
        distance_cost_embed = self.distance_cost_embed(agent_state[:,:,2:4])
        next_cost_embed = self.next_cost_embed(agent_state[:,:,4:7])
        problem_scale_embed = self.problem_scale_embed(agent_state[:,:,7:8])
        global_graph_embed = self.graph_embed(graph_embed).expand_as(depot_pos_embed)

        position_embed = self.position_embed(agent_state.size(1))[None, :].expand_as(depot_pos_embed)
        context =global_graph_embed + depot_pos_embed + distance_cost_embed + next_cost_embed + problem_scale_embed + position_embed
        # context = torch.cat([global_graph_embed, depot_pos_embed + distance_cost_embed + next_cost_embed + problem_scale_embed + position_embed ], dim=-1)
        agent_embed = self.agent_embed(context)
        return agent_embed

class AgentAttentionEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2, norm = "batch"):
        super(AgentAttentionEncoder, self).__init__()
        self.agent_embed = AgentEmbedding(input_dim, hidden_dim, embed_dim)
        self.agent_self_att = nn.ModuleList(
            [
                MultiHeadAttentionLayer(embed_dim, embed_dim, hidden_dim, n_heads=num_heads, norm=norm)
                for _ in range(num_layers)
            ]
        )
        self.num_heads = num_heads

    def forward(self,cities_embed, graph, agent, masks = None):
        """
        :param agent: [B,N,2]
        :return:
        """
        agent_embed = self.agent_embed(cities_embed, graph, agent)
        for model in self.agent_self_att:
            agent_embed = model(agent_embed, mask = masks)
        return agent_embed

if __name__ == '__main__':
    from CityAttentionEncoder import CityAttentionEncoder
    encoder = CityAttentionEncoder(4, 128, 2, 2, 'batch', 256)
    city = torch.randn((2, 10, 2))
    city_mask = torch.randint(0, 2, (2, 10, 10)).bool()
    city_embed, city_embed_mean = encoder(city, city_mask)

    agent_encoder =AgentAttentionEncoder(input_dim=2, hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2, norm='batch')
    agent1 = torch.randint(0,10,(2,3,2))
    agent2 = torch.randn((2,3,8))
    agents_mask = torch.randint(0, 2, (2, 3, 3)).bool()
    agents_embed = agent_encoder(city_embed, city_embed_mean, torch.cat([agent1, agent2], dim=-1), agents_mask)
