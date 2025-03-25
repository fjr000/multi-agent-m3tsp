import argparse

import numpy as np
from torch import inference_mode
from torch.distributions import Categorical

from model.Base.Net import MultiHeadAttentionLayer, SingleHeadAttention, CrossAttentionLayer
from model.n4Model import config
from model.nModel.model_v1 import CityEncoder
from model.Base.Net import CrossAttentionLayer, SingleHeadAttention
from model.Base.Net import SkipConnection, MultiHeadAttention
import torch
import torch.nn as nn
from model.SeqModel.config import Config
from model.Base.Net import initialize_weights
import math


class CityEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(CityEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depot_embed = nn.Linear(self.input_dim, self.embed_dim)
        self.city_embed = nn.Linear(self.input_dim, self.embed_dim)

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
        self.city_embed_mean = None
        self.mean_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.depot_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.node_projection = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, city, city_mask=None):
        """
        :param city: [B,N,2]
        :return:
        """

        if city_mask is not None:
            city_mask[:, 0] = False
        #     B, A = city_mask.shape
        #     expand_masks = city_mask.unsqueeze(1).unsqueeze(1).expand(B, self.num_heads, A, A).reshape(B * self.num_heads, A, A)
        #     # expand_masks.diagonal(dim1=-2, dim2=-1).fill_(False)
        # else:
        #     expand_masks = None

        city_embed = self.city_embed(city)
        for model in self.city_self_att:
            city_embed = model(city_embed, key_padding_mask=city_mask)

        mean = city_embed.mean(dim=1, keepdim=True)
        projection = self.mean_projection(mean)

        depot = self.depot_projection(city_embed[:, 0:1, :])
        nodes = self.node_projection(city_embed[:, 1:, :])

        # del expand_masks
        return city_embed, projection, torch.cat([depot, nodes], dim=1)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=1000):
        super().__init__()
        self.d_model = d_model

        # 创建位置编码
        position_encoding = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('position_encoding', position_encoding)
        self.requires_grad_(False)

    def forward(self, seq_len):
        # positions: [batch_size, seq_len]
        return self.position_encoding[:seq_len, :]


class AgentEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(AgentEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.distance_cost_embed = nn.Linear(2, self.embed_dim, bias=False)
        self.next_cost_embed = nn.Linear(3, self.embed_dim, bias=False)
        self.problem_scale_embed = nn.Linear(1, self.embed_dim, bias=True)

        self.position_embed = PositionalEncoder(self.embed_dim)

    def forward(self, cities_embed, graph_embed, agent_state):
        """
        :param graph_embed
        :param agent_state: [B,M,14]
        :return:
        """
        # cities_expand = cities_embed.expand(agent_state.size(0), -1, -1)
        depot_pos = cities_embed[torch.arange(agent_state.size(0))[:, None, None], agent_state[:, :, :2].long(), :]
        depot_pos_embed = depot_pos[:, :, 0, :] + depot_pos[:, :, 1, :]
        distance_cost_embed = self.distance_cost_embed(agent_state[:, :, 2:4])
        next_cost_embed = self.next_cost_embed(agent_state[:, :, 4:7])
        problem_scale_embed = self.problem_scale_embed(agent_state[:, :, 7:8])
        position_embed = self.position_embed(agent_state.size(1))[None, :]
        agent_embed = graph_embed + depot_pos_embed + distance_cost_embed + next_cost_embed + problem_scale_embed + position_embed
        # agent_embed = graph_embed + depot_pos_embed + distance_cost_embed + next_cost_embed + problem_scale_embed
        # agent_embed = graph_embed + depot_pos_embed + distance_cost_embed + next_cost_embed
        return agent_embed


class AgentSelfEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2, dropout=0):
        super(AgentSelfEncoder, self).__init__()
        self.agent_embed = AgentEmbedding(input_dim, hidden_dim, embed_dim)
        self.agent_self_att = nn.ModuleList(
            [
                MultiHeadAttentionLayer(num_heads, embed_dim, hidden_dim, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.num_heads = num_heads

    def forward(self, cities_embed, graph, agent, masks=None):
        """
        :param agent: [B,N,2]
        :return:
        """
        agent_embed = self.agent_embed(cities_embed, graph, agent)
        if masks is not None:
            expand_masks = masks.unsqueeze(1).expand(masks.size(0), self.num_heads, masks.size(1),
                                                     masks.size(2)).reshape(masks.size(0) * self.num_heads,
                                                                            masks.size(1), masks.size(2))
        else:
            expand_masks = None
        for model in self.agent_self_att:
            agent_embed = model(agent_embed, masks=expand_masks)
        del expand_masks
        return agent_embed


class AgentCityEncoder(nn.Module):
    def __init__(self, hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2, dropout=0):
        super(AgentCityEncoder, self).__init__()
        self.agent_city_att = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads, use_FFN=True, hidden_size=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        # self.linear_forward = nn.Linear(embed_dim, embed_dim)
        # self.action = SingleHeadAttention(embed_dim)
        self.num_heads = num_heads
        self.agent_embed = None

    def forward(self, agent_embed, city_embed, masks):
        expand_masks = masks.unsqueeze(1).expand(agent_embed.size(0), self.num_heads, -1, -1).reshape(
            agent_embed.size(0) * self.num_heads, masks.size(-2), masks.size(-1))
        aca = agent_embed
        for model in self.agent_city_att:
            aca = model(aca, city_embed, city_embed, expand_masks)
        # cross_out = self.linear_forward(aca)
        # action_logits = self.action(aca, city_embed, masks)
        del masks, expand_masks
        #
        # agent_embed_shape = aca.shape
        # agent_embed_reshape = aca.reshape(-1, 1, aca.size(-1))
        # agent_embed_reshape, self.rnn_state = self.rnn(agent_embed_reshape, self.rnn_state)
        # aca = agent_embed_reshape.reshape(*agent_embed_shape)
        self.agent_embed = aca

        return self.agent_embed


class Encoder(nn.Module):
    def __init__(self, config: Config):
        super(Encoder, self).__init__()
        self.config = config
        self.city_encoder = CityEncoder(2, config.city_encoder_hidden_dim, config.embed_dim,
                                        config.city_encoder_num_heads, config.city_encoder_num_layers,
                                        )
        self.agent_encoder = AgentSelfEncoder(config.agent_dim, config.agent_encoder_hidden_dim, config.embed_dim,
                                              config.agent_encoder_num_heads, config.agent_encoder_num_layers,
                                              dropout=config.dropout
                                              )

        self.agent_city_encoder = AgentCityEncoder(config.action_decoder_hidden_dim, config.embed_dim,
                                                   config.action_decoder_num_heads, config.action_decoder_num_layers,
                                                   config.dropout)
        self.value = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(config.embed_dim // 2, 1),
        )

        self.city = None
        self.city_embed = None
        self.city_embed_mean = None
        self.nodes_embed = None

    def init_city(self, city):
        """
        :param city: [B,N,2]
        :return: None
        """
        self.city = city
        self.city_embed, self.city_embed_mean, self.nodes_embed = self.city_encoder(city)

    def forward(self, agent_states, agents_self_mask=None, agents_city_mask=None):
        agent_embed = self.agent_encoder(self.nodes_embed, self.city_embed_mean, agent_states, agents_self_mask)
        agent_embed = self.agent_city_encoder(agent_embed, self.city_embed, agents_city_mask)
        value = self.value(agent_embed)

        return agent_embed, value


class ActionDecoderBlock(nn.Module):
    def __init__(self, config: Config):
        super(ActionDecoderBlock, self).__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.n_heads = config.action_num_heads
        self.hidden_dim = config.action_hidden_dim

        self.self_attn = MultiHeadAttention(
            self.embed_dim,
            self.n_heads,
            dropout=config.dropout
        )
        self.norm = nn.LayerNorm(self.embed_dim)

        self.cross_attn = CrossAttentionLayer(
            self.embed_dim,
            self.n_heads,
            True,
            self.hidden_dim,
            dropout=config.dropout
        )

    def forward(self, action_embed, agent_embed, attn_mask):
        kv = self.self_attn(action_embed, attn_mask=attn_mask)
        kv = self.norm(action_embed + kv)

        embed = self.cross_attn(agent_embed, kv, kv, attn_mask)

        return embed


class ActionsDecoder(nn.Module):
    def __init__(self, config: Config):
        super(ActionsDecoder, self).__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.n_heads = config.action_num_heads
        self.n_layers = config.action_num_layers

        self.act_embed = nn.Linear(self.embed_dim, self.embed_dim)

        self.decoders = nn.ModuleList(
            [ActionDecoderBlock(config)
             for _ in range(self.n_layers)]
        )
        self.act = SingleHeadAttention(self.embed_dim)

    def forward(self, action_embed, agent_embed, attn_mask, city_embed, city_mask, idx=None):
        embed = self.act_embed(action_embed)
        expand_mask = attn_mask[:, None].expand(-1, self.n_heads, -1, -1).reshape(-1, attn_mask.size(-2),
                                                                                  attn_mask.size(-1))
        for model in self.decoders:
            embed = model(embed, agent_embed, expand_mask)

        if idx is not None:
            act_logits = self.act(embed[:, idx:idx + 1, :], city_embed, city_mask)
        else:
            act_logits = self.act(embed, city_embed, city_mask)
        return act_logits


class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.encoder = Encoder(config)
        self.decoder = ActionsDecoder(config)

    def init_city(self, city):
        self.encoder.init_city(city)

    def autoregressive_forward(self, agent_states, salesmen_mask=None, mode="greedy"):
        # seq
        B, A, _ = agent_states.shape

        agent_embed, V = self.encoder(agent_states, agents_city_mask=salesmen_mask)

        cur_pos = self.encoder.city_embed[torch.arange(agent_states.size(0))[:, None], agent_states[:, :,1].long(), :]
        cur_pos_mean = cur_pos.mean(dim = 1)
        actions_embed = torch.zeros((B, A, self.embed_dim), dtype=torch.float32, device=agent_states.device)
        actions_embed[:, 0, :] = cur_pos_mean
        total_act_logits = []
        total_act = []
        totoal_mask = []
        agents_mask = torch.triu(torch.ones(A, A), diagonal=1).to(agent_embed.device).bool()[None,].expand(B, A, A)
        batch_indice = torch.arange(B, device=agent_states.device)[:, None]
        for idx in range(A):
            salesman_mask = salesmen_mask[:, idx:idx + 1, ]
            act_logits = self.decoder(actions_embed[:, :idx + 1], agent_embed[:, :idx + 1],
                                      agents_mask[:, :idx + 1, :idx + 1], self.encoder.city_embed, salesman_mask,
                                      idx=idx)
            total_act_logits.append(act_logits)

            if mode == "greedy":
                act = torch.argmax(act_logits, dim=-1)
            else:
                dist = Categorical(logits=act_logits)
                act = dist.sample()
            total_act.append(act)
            totoal_mask.append(salesman_mask.clone())

            # 提取城市
            if idx < A - 1:
                selected_city = self.encoder.city_embed[batch_indice, act]
                actions_embed[:, idx + 1:idx + 2, :] = selected_city
                salesmen_mask.scatter_(dim=2, index=act.unsqueeze(1).expand(-1, A, -1), value=True)

            # 保证一定有城市可选
            salesmen_mask[torch.argwhere(torch.any(torch.all(salesmen_mask, dim=-1), dim=-1)), :, 0] = False

        act_logits = torch.cat(total_act_logits, dim=1)
        act = torch.cat(total_act, dim=1)
        act_mask = torch.cat(totoal_mask, dim=1)
        return act_logits, act, act_mask, V.squeeze(-1)

    def parallel_forward(self, batch_graph, agent_states, act, salesmen_mask=None):
        self.init_city(batch_graph)

        B, A, _ = agent_states.shape
        # N = batch_graph.size(1)
        #
        # self.encoder.city_embed = self.encoder.city_embed[None, :].expand(expand_step, -1, -1, -1).reshape(B, N, -1)
        # self.encoder.nodes_embed = self.encoder.nodes_embed[None, :].expand(expand_step, -1, -1, -1).reshape(B, N, -1)
        # self.encoder.city_embed_mean = self.encoder.city_embed_mean[None, :].expand(expand_step, -1, -1, -1).reshape(B, 1,
        #                                                                                                           -1)

        agent_embed, V = self.encoder(agent_states, agents_city_mask=salesmen_mask)

        batch_indice = torch.arange(B, device=agent_states.device)[:, None]
        cur_pos = self.encoder.city_embed[torch.arange(agent_states.size(0))[:, None], agent_states[:, :,1].long(), :]
        cur_pos_mean = cur_pos.mean(dim = 1)
        actions_embed = torch.zeros((B, A, self.embed_dim), dtype=torch.float32, device=agent_states.device)
        actions_embed[:, 0, :] = cur_pos_mean
        actions_embed[:,1:,:] = self.encoder.city_embed[batch_indice, act[..., :-1]]

        agents_mask = torch.triu(torch.ones(A, A), diagonal=1).to(agent_embed.device).bool()[None,].expand(B, A, A)
        act_logits = self.decoder(actions_embed, agent_embed,
                                  agents_mask, self.encoder.city_embed, salesmen_mask)

        return act_logits, V.squeeze(-1)

    def forward(self, agent_states, salesmen_mask=None, mode="greedy", act=None, batch_graph=None):
        if act is None:
            return self.autoregressive_forward(agent_states, salesmen_mask, mode)
        else:
            return self.parallel_forward(batch_graph, agent_states, act, salesmen_mask)

    def get_value(self, agent_states, salesmen_mask):
        agent_embed, V = self.encoder(agent_states, agents_city_mask=salesmen_mask)
        return V.squeeze(-1)


if __name__ == "__main__":
    from envs.MTSP.MTSP5 import MTSPEnv
    from utils.TensorTools import _convert_tensor

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--agent_num", type=int, default=10)
    parser.add_argument("--fixed_agent_num", type=bool, default=False)
    parser.add_argument("--agent_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_max_norm", type=float, default=1)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=1e-2)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--city_nums", type=int, default=50)
    parser.add_argument("--random_city_num", type=bool, default=True)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=140000)
    parser.add_argument("--env_masks_mode", type=int, default=3,
                        help="0 for only the min cost  not allow back depot; 1 for only the max cost allow back depot")
    parser.add_argument("--eval_interval", type=int, default=400, help="eval  interval")
    parser.add_argument("--use_conflict_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_conflict_model", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--train_actions_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_city_encoder", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--use_agents_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--use_city_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--agents_adv_rate", type=float, default=0.1, help="rate of adv between agents")
    parser.add_argument("--conflict_loss_rate", type=float, default=0.1, help="rate of adv between agents")
    parser.add_argument("--only_one_instance", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--save_model_interval", type=int, default=10000, help="save model interval")
    parser.add_argument("--seed", type=int, default=528, help="random seed")
    args = parser.parse_args()

    env = MTSPEnv({
        "env_masks_mode": args.env_masks_mode,
        "use_conflict_model": args.use_conflict_model
    })

    from envs.GraphGenerator import GraphGenerator as GG

    graphG = GG(args.batch_size, args.city_nums, 2)
    agent_num = args.agent_num
    model = Model(Config)
    device = torch.device("cuda:0")
    model.to(device)
    episode_logits_list = []
    episode_V_list = []
    episode_mask_list = []
    episode_graph_list = []
    episode_act_list = []
    episode_states_list = []
    episode_step_list = []
    for _ in range(2):

        batch_graph = graphG.generate()
        states, env_info = env.reset(
            config={
                "cities": batch_graph.shape[1],
                "salesmen": agent_num,
                "mode": "fixed",
                "N_aug": batch_graph.shape[0],
                "use_conflict_model": args.use_conflict_model,
            },
            graph=batch_graph
        )

        salesmen_masks = env_info["salesmen_masks"]
        masks_in_salesmen = env_info["masks_in_salesmen"]
        city_mask = env_info["mask"]
        global_mask = env_info["mask"]

        done = False
        use_conflict_model = False

        batch_graph_t = _convert_tensor(batch_graph, device=device)
        model.init_city(batch_graph_t)

        logits_buffer_list = []
        act_buffer_list = []
        mask_buffer_list = []
        V_buffer_list = []
        states_buffer_list = []
        episode_step = 0
        while not done:
            states_t = _convert_tensor(states, device=device)
            # mask: true :not allow  false:allow

            salesmen_masks_t = _convert_tensor(~salesmen_masks, dtype=torch.bool, device=device)
            masks_in_salesmen_t = _convert_tensor(~masks_in_salesmen, dtype=torch.bool, device=device)
            city_mask_t = _convert_tensor(~city_mask, dtype=torch.bool, device=device)

            logits, act, mask, V = model(states_t, salesmen_mask=salesmen_masks_t, mode="sample")
            act_np = act.cpu().numpy();
            states, r, done, env_info = env.step(act_np + 1)
            salesmen_masks = env_info["salesmen_masks"]
            masks_in_salesmen = env_info["masks_in_salesmen"]
            city_mask = env_info["mask"]

            logits_buffer_list.append(logits)
            act_buffer_list.append(act)
            mask_buffer_list.append(mask)
            V_buffer_list.append(V)
            states_buffer_list.append(states_t)
            episode_step += 1

        episode_states_list.append(torch.cat(states_buffer_list, dim=0))
        episode_act_list.append(torch.cat(act_buffer_list, dim=0))
        episode_mask_list.append(torch.cat(mask_buffer_list, dim=0))
        episode_V_list.append(torch.cat(V_buffer_list, dim=0))
        episode_logits_list.append(torch.cat(logits_buffer_list, dim=0))
        episode_graph_list.append(batch_graph_t)
        episode_step_list.append(episode_step)

    from utils.GraphPlot import GraphPlot as GP

    for i in range(2):
        nlogits, nV = model(
            episode_states_list[i],
            episode_mask_list[i],
            act=episode_act_list[i],
            batch_graph=episode_graph_list[i],
            expand_step=episode_step_list[i]
        )
    #
    # traj = env_info["trajectories"]
    # traj_list = env.compress_adjacent_duplicates_optimized(traj)
    # print(env.check_array_structure())
    # gp = GP()
    # gp.draw_route(episode_graph_list[0][0], traj_list[0], one_first=True)
    # gp.draw_route(graph, EndInfo["trajectories"], title="random", one_first=True)
