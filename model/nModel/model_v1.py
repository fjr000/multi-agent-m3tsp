import argparse

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
        self.depot_embed = nn.Linear(self.input_dim, self.embed_dim)
        self.city_embed = nn.Linear(self.input_dim, self.embed_dim)
        # self.depot_embed = nn.Sequential(
        #     nn.Linear(self.input_dim, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, self.embed_dim),
        # )
        #
        # self.city_embed = nn.Sequential(
        #     nn.Linear(self.input_dim, self.embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.embed_dim, self.embed_dim),
        # )

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
        self.city_self_att = nn.ModuleList(
            [
                MultiHeadAttentionLayer(num_heads, embed_dim, hidden_dim)
                for _ in range(num_layers)
            ]
        )
        self.num_heads = num_heads

    def forward(self, city, city_mask=None):
        """
        :param city: [B,N,2]
        :return:
        """

        if city_mask is not None:
            city_mask[:,0] = False
            B, A = city_mask.shape
            expand_masks = city_mask.unsqueeze(1).unsqueeze(1).expand(B, self.num_heads, A, A).reshape(B * self.num_heads, A, A)
            # expand_masks.diagonal(dim1=-2, dim2=-1).fill_(False)
        else:
            expand_masks = None

        city_embed = self.city_embed(city)
        for model in self.city_self_att:
            city_embed = model(city_embed, expand_masks)
        del expand_masks
        return city_embed

class AgentEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(AgentEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.depot_embed = nn.Linear(2, self.embed_dim)
        self.cur_city_embed = nn.Linear(2, self.embed_dim)
        self.distance_embed = nn.Linear(2, self.embed_dim)
        self.graph_embed = nn.Linear(self.embed_dim, self.embed_dim)

        self.agent_embed = nn.Linear(4 * self.embed_dim, self.embed_dim)

    def forward(self, graph_embed, agent_state):
        """
        :param graph_embed
        :param agent_state: [B,M,2]
        :return:
        """
        global_graph_embed = self.graph_embed(graph_embed)
        depot_embed = self.depot_embed(agent_state[:,:,0:2])
        cur_city_embed = self.cur_city_embed(agent_state[:,:,2:4])
        distance_embed = self.distance_embed(agent_state[:,:,4:6])
        context = torch.cat([depot_embed, cur_city_embed, distance_embed, global_graph_embed], dim=-1)
        agent_embed = self.agent_embed(context)
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

    def forward(self, graph, agent):
        """
        :param agent: [B,N,2]
        :return:
        """
        agent_embed = self.agent_embed(graph, agent)
        agent_self_att = self.agent_self_att(agent_embed)
        return agent_self_att

class ActionDecoder(nn.Module):
    def __init__(self, hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2):
        super(ActionDecoder, self).__init__()
        self.agent_city_att = CrossAttentionLayer(embed_dim, num_heads, use_FFN=True, hidden_size=hidden_dim)
        # self.linear_forward = nn.Linear(embed_dim, embed_dim)
        self.action = SingleHeadAttention(embed_dim)
        self.num_heads = num_heads

    def forward(self, agent_embed, city_embed, masks):
        expand_city_embed = city_embed.expand(agent_embed.size(0), -1, -1)
        expand_masks = masks.unsqueeze(1).expand(agent_embed.size(0), self.num_heads, -1, -1)
        expand_masks = expand_masks.reshape(-1, expand_masks.size(2), expand_masks.size(3))
        aca = self.agent_city_att(agent_embed, expand_city_embed, expand_city_embed, expand_masks)
        # cross_out = self.linear_forward(aca)
        action_logits = self.action(aca, expand_city_embed, masks)
        return action_logits


class Model(nn.Module):
    def __init__(self, agent_dim=6, hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2):
        super(Model, self).__init__()
        self.city_encoder = CityEncoder(2, hidden_dim, embed_dim, num_heads, num_layers)
        self.agent_encoder = AgentEncoder(agent_dim, hidden_dim, embed_dim, num_heads, num_layers)
        self.agent_decoder = ActionDecoder(hidden_dim, embed_dim, num_heads, num_layers)
        # self.action_reselector = ActionReselector(embed_dim, num_heads, num_layers)
        self.city_embed = None

    def init_city(self, city):
        """
        :param city: [B,N,2]
        :return: None
        """
        self.city_embed = self.city_encoder(city)

    def forward(self, agent, mask):
        graph = torch.mean(self.city_embed, dim=1)
        expand_graph = graph.unsqueeze(0).expand(agent.size(0), agent.size(1), -1)
        agent_embed = self.agent_encoder(expand_graph, agent)
        actions_logits = self.agent_decoder(agent_embed, self.city_embed, mask)

        # expanded_city_embed = self.city_embed.expand(select.size(1), -1, -1)
        # expanded_select = select.unsqueeze(-1).expand(-1,-1,128)
        # select_city_embed = torch.gather(expanded_city_embed,1, expanded_select)
        # reselect = self.action_reselector(agent_embed, select_city_embed)
        return actions_logits


if __name__ == "__main__":
    from envs.MTSP.MTSP2 import MTSPEnv
    from utils.TensorTools import _convert_tensor

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_worker", type=int, default=4)
    parser.add_argument("--agent_num", type=int, default=5)
    parser.add_argument("--agent_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad_max_norm", type=float, default=0.5)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--returns_norm", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=float, default=512)
    parser.add_argument("--city_nums", type=int, default=50)
    parser.add_argument("--allow_back", type=bool, default=False)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=0)
    args = parser.parse_args()

    cfg = {
        "salesmen": args.agent_num,
        "cities": args.city_nums,
        "seed": None,
        "mode": 'rand'
    }
    env = MTSPEnv(
        cfg
    )
    states, info = env.reset()
    anum = info["salesmen"]
    cnum = info["cities"]
    graph = info["graph"]
    global_mask = info["mask"]
    agents_mask = info["mask"][np.newaxis].repeat(anum, axis=0)

    done = False
    EndInfo = {}
    EndInfo.update(info)
    agent_config = {
        "city_nums": cnum,
        "agent_nums": anum,
    }
    graph_t = _convert_tensor(graph, device="cuda", target_shape_dim=3)
    from algorithm.DNN2.AgentBase import AgentBase

    agent = AgentBase(args, Model)
    agent.reset_graph(graph_t)

    states_nb, actions_nb, returns_nb, masks_nb, done_nb = (
        agent._run_episode(env, graph[np.newaxis,], anum, eval_mode=False, exploit_mode="greedy"))

    from utils.GraphPlot import GraphPlot as GP

    gp = GP()
    # gp.draw_route(graph, EndInfo["trajectories"], title="random", one_first=True)
