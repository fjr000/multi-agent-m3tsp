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

        self.depot_pos_embed = nn.Linear(2 * self.embed_dim, self.embed_dim, bias=False)
        self.distance_cost_embed = nn.Linear(6, self.embed_dim, bias= False)
        self.graph_embed = nn.Linear(self.embed_dim, self.embed_dim, bias= False)

        # self.position_embed = PositionalEncoder(self.embed_dim)
        #
        # self.agent_embed = nn.Sequential(
        #     nn.Linear(3 * self.embed_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, self.embed_dim),
        # )

    def forward(self, cities_embed, n_depot_embed, graph_embed, agent_state):
        """
        :param graph_embed
        :param agent_state: [B,M,14]
        :return:
        """

        # cities_expand = cities_embed.expand(agent_state.size(0), -1, -1)

        cur_pos = cities_embed[torch.arange(agent_state.size(0))[:, None], agent_state[:, :, 1].long(),:]
        depot_pos = torch.cat([n_depot_embed, cur_pos], dim=-1)
        depot_pos_embed = self.depot_pos_embed(depot_pos)
        distance_cost_embed = self.distance_cost_embed(agent_state[:, :, 2:8])
        global_graph_embed = self.graph_embed(graph_embed)

        context = depot_pos_embed + distance_cost_embed + global_graph_embed

        agent_embed = context
        return agent_embed


class AgentAttentionEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2, norm="batch"):
        super(AgentAttentionEncoder, self).__init__()
        self.agent_embed = AgentEmbedding(input_dim, hidden_dim, embed_dim)
        self.agent_self_att = nn.ModuleList(
            [
                MultiHeadAttentionLayer(embed_dim, embed_dim, hidden_dim, n_heads=num_heads, norm=norm)
                for _ in range(num_layers)
            ]
        )
        self.num_heads = num_heads

    def forward(self, cities_embed, n_depot_embed, graph, agent, masks=None):
        """
        :param agent: [B,N,2]
        :return:
        """
        agent_embed = self.agent_embed(cities_embed, n_depot_embed, graph, agent)
        for model in self.agent_self_att:
            agent_embed = model(agent_embed, mask=masks)
        return agent_embed


class AgentAttentionRNNEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2, norm="batch", rnn_type='GRU'):
        super(AgentAttentionRNNEncoder, self).__init__()
        self.agent_embed = AgentEmbedding(input_dim, hidden_dim, embed_dim)
        self.agent_self_att = nn.ModuleList(
            [
                MultiHeadAttentionLayer(embed_dim, embed_dim, hidden_dim, n_heads=num_heads, norm=norm)
                for _ in range(num_layers)
            ]
        )

        self.rnn = None
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=embed_dim,
                hidden_size=embed_dim,
                batch_first=True,
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=embed_dim,
                hidden_size=embed_dim,
                batch_first=True,
            )
        else:
            raise NotImplementedError

        self.num_heads = num_heads
        self.rnn_state = None

    def init_rnn_state(self, batch_size, agent_num, device):
        if isinstance(self.rnn, nn.GRU):
            self.rnn_state = torch.zeros(
                (1, batch_size * agent_num, self.rnn.hidden_size),
                dtype=torch.float32,
                device=device
            )
        elif isinstance(self.rnn, nn.LSTM):
            self.rnn_state = [torch.zeros(
                (1, batch_size * agent_num, self.rnn.hidden_size),
                dtype=torch.float32,
                device=device
            ),
                torch.zeros(
                    (1, batch_size * agent_num, self.rnn.hidden_size),
                    dtype=torch.float32,
                    device=device
                )
            ]
        else:
            raise NotImplementedError

    def forward(self, cities_embed, graph, agent, masks=None):
        """
        :param agent: [B,N,2]
        :return:
        """
        agent_embed = self.agent_embed(cities_embed, graph, agent)
        for model in self.agent_self_att:
            agent_embed = model(agent_embed, mask=masks)
        agent_embed_shape = agent_embed.shape
        agent_embed_reshape = agent_embed.reshape(-1, 1, agent_embed.size(-1))
        agent_embed_reshape, self.rnn_state = self.rnn(agent_embed_reshape, self.rnn_state)
        agent_embed = agent_embed_reshape.reshape(*agent_embed_shape)

        return agent_embed


if __name__ == '__main__':
    from CityAttentionEncoder import CityAttentionEncoder

    encoder = CityAttentionEncoder(4, 128, 2, 2, 'batch', 256)
    city = torch.randn((2, 10, 2))
    city_mask = torch.randint(0, 2, (2, 10, 10)).bool()
    city_embed, city_embed_mean = encoder(city, city_mask)

    agent_encoder = AgentAttentionEncoder(input_dim=2, hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2,
                                          norm='batch')
    agent1 = torch.randint(0, 10, (2, 3, 2))
    agent2 = torch.randn((2, 3, 8))
    agents_mask = torch.randint(0, 2, (2, 3, 3)).bool()
    agents_embed = agent_encoder(city_embed, city_embed_mean, torch.cat([agent1, agent2], dim=-1), agents_mask)
