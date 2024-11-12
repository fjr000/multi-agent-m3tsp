import numpy as np
import torch
import torch.nn as nn

from model.NNN.ActionReSelector import ActionReSelectorBySeq as ARS
import model.Base.Net as Net


class TestModel(nn.Module):
    def __init__(self, agent_state_dim, city_state_dim, embedding_dim=128):
        super(TestModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.city_state_dim = city_state_dim
        self.agent_state_dim = agent_state_dim

        self.agents_encoder = Net.SelfEncoder([self.agent_state_dim], self.embedding_dim, 8)
        self.cities_encoder = Net.SelfEncoder([self.city_state_dim for _ in range(4)], self.embedding_dim, 8)

        self.cities_embedding = None
        self.graph = None

        self.ARS = ARS()
        self.decoder = Net.SingleDecoder(self.embedding_dim, 8)

    def initGraph(self, graph, depot_idx=1):
        self.graph = graph
        self.cities_embedding = self.cities_encoder()

    def forward(self, agent_state, city_state, index):
        assert agent_state[0].shape[
                   -1] == self.agent_state_dim, f"the dimension of input agent state is {agent_state[0].shape[-1]} and should be equal to {self.agent_state_dim} "
        assert city_state[0].shape[
                   -1] == self.city_state_dim, f"the dimension of input agent state is {city_state[0].shape[-1]} and should be equal to {self.city_state_dim} "

        agent_embedding = self.agents_encoder(agent_state)[0]
        city_embeddings = self.cities_encoder(city_state)
        depot_embedding = city_embeddings[0]
        last_visit_embedding = city_embeddings[1]
        cur_visit_embedding = city_embeddings[2]
        to_visit_embedding = city_embeddings[3]
        city_embeddings_cat = torch.cat(city_embeddings, dim=-2)
        graph_embedding = city_embeddings_cat.mean(dim=-2)

        outs = []

        for i in range(agent_embedding.shape[1]):
            single_city_embeddings = torch.cat(
                [depot_embedding, last_visit_embedding[:, i].unsqueeze(1), cur_visit_embedding[:, i].unsqueeze(1),
                 to_visit_embedding], dim=1)
            single_city_idx = index[0]
            single_city_idx.extend(index[1][i])
            single_city_idx.extend(index[2][i])
            single_city_idx.extend(index[3])
            out = self.decoder(agent_embedding[:, i].unsqueeze(1), single_city_embeddings)
            idxx = single_city_idx[out]
            outs.append(idxx)
        return outs

    def reset(self, agent_config):
        pass

    def predict(self, agent_state, city_state, index):
        self.forward(agent_state, city_state, index)


from model.Base.AgentBase import AgentBase


class TestModel2(AgentBase):

    def __init__(self, config):
        super().__init__(config)
        self.embeddings_dim = None
        self.city_state_dim = None
        self.agent_state_dim = None

        self.agents_encoder = Net.SelfEncoder([self.agent_state_dim], self.embeddings_dim, 8)
        self.cities_encoder = Net.SelfEncoder([self.city_state_dim for _ in range(4)], self.embeddings_dim, 8)

        self.cities_embedding = None
        self.graph = None

        self.ARS = ARS()
        self.decoder = Net.SingleDecoder(self.embeddings_dim, 8)

    def reset(self, config):
        super().reset(config)
        self.parse_config(config)

    def parse_config(self, config):
        super().parse_config(config)
        if config is not None:
            embeddings_dim = config.get("embeddings_dim")
            if embeddings_dim is not None:
                self.embeddings_dim = embeddings_dim

            agent_state_dim = config.get("agent_state_dim")
            if agent_state_dim is not None:
                self.agent_state_dim = agent_state_dim

            city_state_dim = config.get("city_state_dim")
            if city_state_dim is not None:
                self.city_state_dim = city_state_dim

    def forward(self, agent_state, city_state, index):
        assert agent_state[0].shape[
                   -1] == self.agent_state_dim, f"the dimension of input agent state is {agent_state[0].shape[-1]} and should be equal to {self.agent_state_dim} "
        assert city_state[0].shape[
                   -1] == self.city_state_dim, f"the dimension of input agent state is {city_state[0].shape[-1]} and should be equal to {self.city_state_dim} "

        agent_embedding = self.agents_encoder(agent_state)[0]
        city_embeddings = self.cities_encoder(city_state)
        depot_embedding = city_embeddings[0]
        last_visit_embedding = city_embeddings[1]
        cur_visit_embedding = city_embeddings[2]
        to_visit_embedding = city_embeddings[3]
        city_embeddings_cat = torch.cat(city_embeddings, dim=-2)
        graph_embedding = city_embeddings_cat.mean(dim=-2)

        outs = []

        for i in range(agent_embedding.shape[1]):
            single_city_embeddings = torch.cat(
                [depot_embedding, last_visit_embedding[:, i].unsqueeze(1), cur_visit_embedding[:, i].unsqueeze(1),
                 to_visit_embedding], dim=1)
            single_city_idx = index[0]
            single_city_idx.extend(index[1][i])
            single_city_idx.extend(index[2][i])
            single_city_idx.extend(index[3])
            out = self.decoder(agent_embedding[:, i].unsqueeze(1), single_city_embeddings)
            idxx = single_city_idx[out]
            outs.append(idxx)
        return outs

    def predict(self, agents_state, cities_state, global_mask, agents_action_mask, mode=None):

        pass

    def save_model(self):
        pass

    def load_model(self):
        pass


if __name__ == '__main__':
    model = TestModel(3, 2, 128)
    from envs.MTSP.MTSP import MTSPEnv as ENV

    Config = {
        "city_nums": (10, 100),
        "agent_nums": (1, 9),
        "seed": 1111,
        "fixed_graph": False,
        "allow_back": True
    }
    env = ENV(Config)
    states, info = env.reset()
    anum = info.get("anum")
    cnum = info.get("cnum")
    graph = info.get("graph")
    graph_matrix = info.get("graph_matrix")
    global_mask = info.get("global_mask")
    agents_action_mask = info.get("agents_action_mask")


    def get_cur_idxs(agents_trajectory):
        return np.array(agents_trajectory[:, -1], dtype=np.int64)


    def get_last_idxs(agents_trajectory):
        last_idxs = []
        for traj in agents_trajectory:
            if len(traj) > 1:
                last_idxs.append(traj[-2])
            else:
                last_idxs.append(traj[-1])
        return np.array(last_idxs, dtype=np.int64)


    def city_parse(graph, global_mask, last_idxs, cur_idxs):
        depot = graph[0]
        last_visited_city = graph[last_idxs - 1]
        cur_visited_city = graph[cur_idxs - 1]

        to_visited_cities_idxs = np.argwhere(global_mask == 0).squeeze(-1)
        to_visited_city = graph[to_visited_cities_idxs]

        return [depot, last_visited_city, cur_visited_city, to_visited_city], [np.ones((1,)), last_idxs, cur_idxs,
                                                                               to_visited_cities_idxs + 1]


    init_city_states, idx = city_parse(graph, global_mask, np.ones(anum, dtype=np.int64), np.ones(anum, dtype=np.int64))
    from utils.TensorTools import _convert_tensor

    states = _convert_tensor(states, target_shape_dim=3)
    init_city_states = _convert_tensor(init_city_states, target_shape_dim=3)
    state = model((states,), init_city_states, [idx])
    print(state)
