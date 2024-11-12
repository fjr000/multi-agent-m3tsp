import torch
import torch.nn as nn
import torch.nn.functional as F
import model.Base.Net as Net
from model.Base.Encoder import Encoder


class TestModel(nn.Module):
    def __init__(self, city_dim, agent_dim, n_heads, embedding_dim, n_layers, hidden_dim, normalization='batch'):
        super(TestModel, self).__init__()
        self.graph_encoder = Encoder(city_dim, n_heads, embedding_dim, n_layers, hidden_dim, normalization)
        self.agent_encoder = Encoder(agent_dim, n_heads, embedding_dim, n_layers, hidden_dim, normalization)

        self.graph_embeds = None,
        self.graph_embeds_mean = None
        self.decoder = Net.SingleHeadAttention(embedding_dim)

    def init_graph(self, graph):
        self.graph_embeds, self.graph_embeds_mean = self.graph_encoder(graph)

    def _get_graph_embeds(self, mask, get_mean=False):
        assert self.graph_embeds is not None
        selected_city = self.graph_embeds[mask == 0]
        if len(selected_city.shape) == 2:
            selected_city = selected_city.unsqueeze(0)
        if get_mean:
            return selected_city, torch.mean(selected_city, dim=-2)
        else:
            return selected_city

    def _get_city_embeds(self, index):
        return self.graph_embeds_mean[index]

    def _get_deopt_embed(self):
        return self.graph_embeds[:, 0]

    def forward(self, agent_state, agent_id, global_mask, action_list):
        """

        :param action_list: [B, 2] :[[last_idx, cur_idx],...]
        :param global_mask: [B, N]
        :param agent_id: [B]
        :param agent_state: [B, M, state_dim]
        :return: logits: [B, N]
        """

        agents_embeds, agents_embeds_mean = self.agent_encoder(agent_state)
        agent_embeds = agents_embeds[:, agent_id].unsqueeze(1)

        city_embeds = self._get_graph_embeds(global_mask)
        city_idxs = torch.argwhere(global_mask == 0)
        depot = self._get_deopt_embed()
        last_cur_city_embeds = self._get_city_embeds(action_list)
        h = torch.cat((depot.unsqueeze(1), last_cur_city_embeds, city_embeds), dim=-2)

        logits = self.decoder(agent_embeds, h)

        logits = F.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob

    def predict(self,agent_state, agent_id, global_mask, action_list):
        return self.forward(agent_state, agent_id, global_mask, action_list)

if __name__ == '__main__':
    model = TestModel(2, 3, 8, 128, 2, 256, 'none')
    from envs.MTSP.MTSP import MTSPEnv as Env
    import numpy as np

    Config = {
        "city_nums": (10, 20),
        "agent_nums": (1, 4),
        "seed": 1111,
        "fixed_graph": False,
        "allow_back": True
    }
    env = Env(Config)
    states, info = env.reset()
    anum = info.get("anum")
    cnum = info.get("cnum")
    graph = info.get("graph")
    graph_matrix = info.get("graph_matrix")
    global_mask = info.get("global_mask")
    agents_action_mask = info.get("agents_action_mask")
    agents_way= info.get("agents_way")

    from utils.TensorTools import _convert_tensor
    done = False
    graph_tensor = _convert_tensor(graph, target_shape_dim=3)
    model.init_graph(graph_tensor)
    while not done:
        states = _convert_tensor(states, target_shape_dim=3)
        global_mask = _convert_tensor(global_mask, target_shape_dim=2)
        agents_action_mask = _convert_tensor(agents_action_mask,target_shape_dim=3)
        agents_way = _convert_tensor(agents_way - 1, dtype=torch.int64,target_shape_dim=3)
        actions = np.zeros(anum)
        for i in range(anum):
            action = model.predict(states, i, global_mask, agents_way[:,i])
            actions[i] = action
        states, reward, done, info = env.step(actions)
        global_mask = info.get("global_mask")
        agents_action_mask = info.get("agents_action_mask")
        agents_way= info.get("agents_way")


