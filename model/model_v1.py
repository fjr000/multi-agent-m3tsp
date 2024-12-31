import numpy as np
import torch
import torch.nn as nn

import model.Base.Net as Net
from model.Base.Net import MultiHeadAttentionLayer, SingleHeadAttention, CrossAttentionLayer


class CityEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(CityEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depot_embed = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.embed_dim),
        )

        self.city_embed = nn.Sequential(
            nn.Linear(self.input_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

    def forward(self, city):
        """
        :param city: [B,N,2]
        :return:
        """
        depot_embed = self.depot_embed(city[:, 0:1])
        city_embed = self.city_embed(city[:, 1:])
        cities_embed = torch.cat([depot_embed, city_embed], dim=1)
        return cities_embed


class CityEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2):
        super(CityEncoder, self).__init__()
        self.city_embed = CityEmbedding(input_dim, hidden_dim, embed_dim)
        self.city_self_att = nn.Sequential(
            *[
                MultiHeadAttentionLayer(num_heads, embed_dim, hidden_dim)
                for _ in range(num_layers)
            ]
        )

    def forward(self, city):
        """
        :param city: [B,N,2]
        :return:
        """
        city_embed = self.city_embed(city)
        city_self_att = self.city_self_att(city_embed)
        return city_self_att


class AgentEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(AgentEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.cur_city_embed = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.embed_dim),
        )

        self.last_city_embed = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.embed_dim),
        )

        self.agent_embed = nn.Sequential(
            nn.Linear(2 * self.embed_dim, self.embed_dim),
        )

    def forward(self, agent_state_list):
        """
        :param agent_state: [[B,M,2],...]
        :return:
        """
        last_city_embed = self.last_city_embed(agent_state_list[0])
        cur_city_embed = self.cur_city_embed(agent_state_list[1])
        city_embed = torch.cat([last_city_embed, cur_city_embed], dim=-1)
        agent_embed = self.agent_embed(city_embed)
        return agent_embed


class AgentEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2):
        super(AgentEncoder, self).__init__()
        self.agent_embed = AgentEmbedding(input_dim, hidden_dim, embed_dim)
        self.agent_self_att = nn.Sequential(
            *[
                MultiHeadAttentionLayer(num_heads, embed_dim, hidden_dim)
                for _ in range(num_layers)
            ]
        )

    def forward(self, agent):
        """
        :param agent: [B,N,2]
        :return:
        """
        agent_embed = self.agent_embed(agent)
        agent_self_att = self.agent_self_att(agent_embed)
        return agent_self_att


class ActionDecoder(nn.Module):
    def __init__(self, hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2):
        super(ActionDecoder, self).__init__()
        self.agent_city_att = CrossAttentionLayer(embed_dim, num_heads, use_FFN=True, hidden_size=hidden_dim)
        self.linear_forward = nn.Linear(embed_dim, embed_dim)
        self.action = SingleHeadAttention(embed_dim)

    def forward(self, agent_embed, city_embed, attn_mask, mode="sample"):
        aca = self.agent_city_att(agent_embed, city_embed, city_embed, attn_mask)
        cross_out = self.linear_forward(aca)
        action_logits = self.action(cross_out, city_embed, attn_mask.unsqueeze(0))
        dist = torch.distributions.Categorical(logits=action_logits)
        if mode == "greedy":
            action = torch.argmax(action_logits, dim=-1)
            logprob = dist.log_prob(action)
        elif mode == "sample":
            action = dist.sample()
            logprob = dist.log_prob(action)
        return action, logprob


class ActionReselector(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=2):
        super(ActionReselector, self).__init__()
        self.act_to_agent = SingleHeadAttention(embed_dim)

    def forward(self, agent_embed, city_embed):
        city_to_agent = self.act_to_agent(city_embed, agent_embed)
        agent = torch.argmax(city_to_agent, dim=-1)
        return agent


class Model(nn.Module):
    def __init__(self, agent_dim=3, hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2):
        super(Model, self).__init__()
        self.city_encoder = CityEncoder(2, hidden_dim, embed_dim, num_heads, num_layers)
        self.agent_encoder = AgentEncoder(agent_dim, hidden_dim, embed_dim, num_heads, num_layers)
        self.agent_decoder = ActionDecoder(hidden_dim, embed_dim, num_heads, num_layers)
        self.action_reselector = ActionReselector(embed_dim, num_heads, num_layers)
        self.city_embed = None

    def init_city(self, city):
        self.city_embed = self.city_encoder(city)

    def forward(self, agent, mask):
        agent_embed = self.agent_encoder(agent)
        select, logprob = self.agent_decoder(agent_embed, self.city_embed, mask)
        # expanded_city_embed = self.city_embed.expand(select.size(1), -1, -1)
        # expanded_select = select.unsqueeze(-1).expand(-1,-1,128)
        # select_city_embed = torch.gather(expanded_city_embed,1, expanded_select)
        # reselect = self.action_reselector(agent_embed, select_city_embed)
        return select + 1, logprob


if __name__ == "__main__":
    from envs.MTSP.MTSP import MTSPEnv
    from utils.TensorTools import _convert_tensor

    env = MTSPEnv()
    cfg = {
        "city_nums": (5, 10),
        "agent_nums": (2, 3),
        "seed": None,
        "fixed_graph": False
    }
    env = MTSPEnv(
        cfg
    )
    states, info = env.reset()
    anum = info["anum"]
    cnum = info["cnum"]
    graph = info["graph"]
    global_mask = info["global_mask"]
    agents_mask = info["agents_mask"]
    agents_last_states = info["actors_last_state"]
    agents_way = info["agents_way"]

    done = False
    EndInfo = {}
    EndInfo.update(info)
    agent_config = {
        "city_nums": cnum,
        "agent_nums": anum,
    }
    graph_t = _convert_tensor(graph, device="cuda", target_shape_dim=3)
    model = Model(agent_dim=3).to("cuda")
    model.init_city(graph_t)
    from model.NNN.RandomAgent import RandomAgent

    agent = RandomAgent()
    reward = 0
    while not done:
        state_t = _convert_tensor(states, device="cuda", target_shape_dim=3)
        last_state_t = _convert_tensor(agents_last_states, device="cuda", target_shape_dim=3)
        agents_mask_t = _convert_tensor(agents_mask, device="cuda", target_shape_dim=2)
        actions, logp = model([last_state_t, state_t], agents_mask_t)
        # action = np.random.choice(action_to_chose, anum, replace=False)

        actions_numpy = actions.squeeze(0).cpu().numpy()
        s = set()
        true_logp = np.zeros_like(actions_numpy)
        for i in range(len(actions_numpy)):
            if actions_numpy[i] == 1 or actions_numpy[i] not in s:
                s.add(actions_numpy[i])
            else:
                actions_numpy[i] = 0
            if actions_numpy[i] == agents_way[i, -1]:
                actions_numpy[i] = 0
            elif actions_numpy[i] == agents_way[i, 0]:
                actions_numpy[i] = -1
        states, reward, done, info = env.step(actions_numpy)
        global_mask = info["global_mask"]
        agents_mask = info["agents_mask"]
        agents_way = info["agents_way"]
        agents_last_states = info["actors_last_state"]
        if done:
            EndInfo.update(info)
    loss = reward
    print(f"reward:{reward}")
    print(f"trajectory:{EndInfo}")
    from utils.GraphPlot import GraphPlot as GP

    gp = GP()
    gp.draw_route(graph, EndInfo["actors_trajectory"], title="random", one_first=True)
