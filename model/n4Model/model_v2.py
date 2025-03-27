import argparse

import numpy as np
from model.Base.Net import MultiHeadAttentionLayer
from model.Base.Net import CrossAttentionLayer, SingleHeadAttention
import torch
import torch.nn as nn
from model.n4Model.config import Config
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
        # cities_embed = torch.cat([depot_embed, city_embed], dim=1)
        return depot_embed, city_embed


class CityEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2):
        super(CityEncoder, self).__init__()
        self.city_embed = CityEmbedding(input_dim, hidden_dim, embed_dim)
        self.city_self_att = nn.ModuleList(
            [
                MultiHeadAttentionLayer(num_heads, embed_dim, hidden_dim, normalization='batch')
                for _ in range(num_layers)
            ]
        )
        self.num_heads = num_heads
        self.city_embed_mean = None
        self.position_encoder = PositionalEncoder(embed_dim)
        self.pos_embed_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, city, n_agents, city_mask=None):
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

        depot_embed, city_embed = self.city_embed(city)

        pos_embed = self.position_encoder(n_agents + 1)
        depot_embed_repeat = depot_embed.expand(-1, n_agents + 1, -1)
        pos_embed = self.alpha * self.pos_embed_proj(pos_embed)
        depot_pos_embed = depot_embed_repeat + pos_embed[None, :]

        graph_embed = torch.cat([depot_pos_embed[:, 0:1, :], city_embed, depot_pos_embed[:, 1:, :]], dim=1)

        for model in self.city_self_att:
            graph_embed = model(graph_embed, key_padding_mask=city_mask)

        # del expand_masks
        return (
            graph_embed,  # (B,A+N,E)
            graph_embed.mean(keepdim=True, dim=1)  # (batch_size, 1, embed_dim) mean(A+E)
        )


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

        self.depot_pos_embed = nn.Linear(2 * self.embed_dim, self.embed_dim, bias=False)
        self.distance_cost_embed = nn.Linear(12, self.embed_dim, bias=False)
        self.graph_embed = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def forward(self, cities_embed, n_depot_embed, graph_embed, agent_state):
        """
        :param graph_embed
        :param agent_state: [B,M,14]
        :return:
        """
        cur_pos = cities_embed[torch.arange(agent_state.size(0))[:, None], agent_state[:, :, 1].long(), :]
        depot_pos = torch.cat([n_depot_embed, cur_pos], dim=-1)
        depot_pos_embed = self.depot_pos_embed(depot_pos)
        distance_cost_embed = self.distance_cost_embed(agent_state[:, :, 2:])
        global_graph_embed = self.graph_embed(graph_embed)

        context = depot_pos_embed + distance_cost_embed + global_graph_embed

        agent_embed = context
        return agent_embed


class AgentEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2, dropout=0):
        super(AgentEncoder, self).__init__()
        self.agent_embed = AgentEmbedding(input_dim, hidden_dim, embed_dim)
        self.agent_self_att = nn.ModuleList(
            [
                MultiHeadAttentionLayer(num_heads, embed_dim, hidden_dim, normalization='layer', dropout=dropout)
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

        n_agents = n_depot_embed.shape[1]
        # 禁掉自注意力
        if n_agents > 1 and masks is not None:
            self_mask = torch.eye(n_agents, dtype=torch.bool, device=graph.device)[None, :].expand_as(masks)  # 对角线为True
        else:
            self_mask = None

        expand_masks = None
        if masks is not None:
            if self_mask is not None:
                masks = self_mask | masks
            expand_masks = masks.unsqueeze(1).expand(masks.size(0), self.num_heads, masks.size(1),
                                                     masks.size(2)).reshape(masks.size(0) * self.num_heads,
                                                                            masks.size(1), masks.size(2))
        else:
            if self_mask is not None:
                masks = self_mask
                expand_masks = masks.unsqueeze(1).expand(masks.size(0), self.num_heads, masks.size(1),
                                                     masks.size(2)).reshape(masks.size(0) * self.num_heads,
                                                                            masks.size(1), masks.size(2))

        for model in self.agent_self_att:
            agent_embed = model(agent_embed, masks=expand_masks)
        del expand_masks
        return agent_embed


class ActionDecoder(nn.Module):
    def __init__(self, hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2, dropout=0, rnn_type='GRU'):
        super(ActionDecoder, self).__init__()
        self.agent_city_att = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads, use_FFN=True, hidden_size=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        # self.linear_forward = nn.Linear(embed_dim, embed_dim)
        self.action = SingleHeadAttention(embed_dim)
        self.num_heads = num_heads

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
        self.agent_embed = None

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

    def forward(self, agent_embed, city_embed, masks):
        # expand_city_embed = city_embed.expand(agent_embed.size(0), -1, -1)
        extra_masks = ~torch.eye(agent_embed.size(1), device=agent_embed.device).bool()[None, :].expand(
            agent_embed.size(0), -1, -1)
        expand_masks = torch.cat([masks, extra_masks], dim=-1)
        expand_masks = expand_masks.unsqueeze(1).expand(agent_embed.size(0), self.num_heads, -1, -1).reshape(
            agent_embed.size(0) * self.num_heads, expand_masks.size(-2), expand_masks.size(-1))
        # expand_masks = expand_masks.reshape(agent_embed.size(0) * self.num_heads, expand_masks.size(2), expand_masks.size(3))
        aca = agent_embed
        for model in self.agent_city_att:
            aca = model(aca, city_embed, city_embed, expand_masks)
        # cross_out = self.linear_forward(aca)
        action_logits = self.action(aca, city_embed[:, :-agent_embed.size(1), :], masks)
        del masks, expand_masks

        agent_embed_shape = aca.shape
        agent_embed_reshape = aca.reshape(-1, 1, aca.size(-1))
        agent_embed_reshape, self.rnn_state = self.rnn(agent_embed_reshape, self.rnn_state)
        aca = agent_embed_reshape.reshape(*agent_embed_shape)
        self.agent_embed = aca

        return action_logits, aca


class ConflictModel(nn.Module):
    def __init__(self, config: Config):
        super(ConflictModel, self).__init__()
        # 不能使用dropout
        self.city_agent_att = nn.ModuleList([
            CrossAttentionLayer(config.embed_dim, config.conflict_deal_num_heads,
                                use_FFN=True, hidden_size=config.conflict_deal_hidden_dim,
                                dropout=0)
            for _ in range(config.conflict_deal_num_layers)
        ])
        # self.linear_forward = nn.Linear(embed_dim, embed_dim)
        self.agents = SingleHeadAttention(config.embed_dim)
        self.num_heads = config.conflict_deal_num_heads

    def forward(self, agent_embed, city_embed, acts, info=None):
        """
        Args:
            agent_embed:   [B,A,E] 智能体特征
            city_embed:    [B,N,E] 城市特征
            acts: [B,A] 动作
        Returns:
            final_cities: [B,A] 最终分配结果
            conflict_mask: [B,N] 初始冲突标记
        """
        B, A, E = agent_embed.shape

        # 2. 生成初始冲突掩码 ----------------------------------------------------
        # 扩展维度用于广播比较
        acts_exp1 = acts.unsqueeze(2)  # [B,5,1]
        acts_exp2 = acts.unsqueeze(1)  # [B,1,5]
        # 生成布尔型冲突矩阵
        conflict_matrix = (acts_exp1 == acts_exp2).bool()  # [B,5,5]
        identity_matrix = torch.eye(A, device=acts.device).unsqueeze(0).bool()  # [1, A, A]
        conflict_matrix = torch.where(acts_exp1 == 0, identity_matrix, conflict_matrix)
        conflict_matrix = ~conflict_matrix
        expand_conflict_mask = conflict_matrix.unsqueeze(1).expand(B, self.num_heads, A, A).reshape(B * self.num_heads,
                                                                                                    A, A)

        # 3. 提取候选城市特征 -----------------------------------------------------
        selected_cities = torch.gather(
            city_embed,
            1,
            acts.unsqueeze(-1).expand(-1, -1, E)
        )  # [B,5,E]

        # 4. 注意力重新分配 ------------------------------------------------------
        # Q: 候选城市特征 [B,5,E]
        # K/V: 智能体特征 [B,5,E]
        cac = selected_cities
        for att in self.city_agent_att:
            cac = att(cac, agent_embed, agent_embed, expand_conflict_mask)

        agents_logits = self.agents(cac, agent_embed, conflict_matrix)

        del conflict_matrix, expand_conflict_mask, identity_matrix, acts_exp1, acts_exp2

        return agents_logits


class ActionsModel(nn.Module):
    def __init__(self, config: Config):
        super(ActionsModel, self).__init__()
        self.city_encoder = CityEncoder(2, config.city_encoder_hidden_dim, config.embed_dim,
                                        config.city_encoder_num_heads, config.city_encoder_num_layers,
                                        )
        self.agent_encoder = AgentEncoder(config.agent_dim, config.agent_encoder_hidden_dim, config.embed_dim,
                                          config.agent_encoder_num_heads, config.agent_encoder_num_layers,
                                          dropout=config.dropout
                                          )
        self.agent_decoder = ActionDecoder(config.action_decoder_hidden_dim, config.embed_dim,
                                           config.action_decoder_num_heads, config.action_decoder_num_layers,
                                           dropout=config.dropout
                                           )
        self.city = None
        self.city_embed = None
        self.city_embed_mean = None
        self.nodes_embed = None
        self.config = config

    def init_city(self, city, n_agents):
        """
        :param city: [B,N,2]
        :return: None
        """
        self.city = city
        self.city_embed, self.city_embed_mean = self.city_encoder(city, n_agents)

    def forward(self, agent, mask, info=None):
        # batch_mask = mask[:,0,:].unsqueeze(-1).expand(mask.size(0),mask.size(2),self.city_embed.shape[-1])
        # ori_expand_graph = self.city_embed.expand(*batch_mask.shape)
        # mask_expand_graph = ori_expand_graph * batch_mask
        # mask_sum_expand_graph = mask_expand_graph.sum(1)
        # non_zero_count = batch_mask.sum(1)
        # avg_graph = mask_sum_expand_graph / non_zero_count

        # expand_graph = self.city_embed_mean.unsqueeze(1).expand(agent.size(0), agent.size(1), -1)
        # expand_graph = self.city_embed_mean.unsqueeze(1)
        # city_mask = None if info is None else info.get("mask", None)
        # self.city_embed = self.city_encoder(self.city, city_mask = city_mask)
        # cnt = torch.count_nonzero(~city_mask, dim=-1).unsqueeze(-1)
        # self.city_embed_mean = torch.sum(self.city_embed, dim=1) / cnt
        # del city_mask
        n_agents = agent.size(1)
        agent_embed = self.agent_encoder(self.city_embed[:, :-n_agents, :], self.city_embed[:, -n_agents:, :],
                                         self.city_embed_mean, agent,
                                         None if info is None else info.get("masks_in_salesmen", None))

        actions_logits, agent_embed = self.agent_decoder(agent_embed, self.city_embed, mask)

        x = torch.all(torch.isinf(actions_logits), dim=-1)
        xxx = torch.isnan(actions_logits).any().item()
        xx = x.any().item()
        if xx or xxx:
            a = torch.argwhere(torch.isnan(actions_logits))
            pass
        a = torch.argwhere(x)

        del mask

        # agents_logits = self.deal_conflict(agent_embed, self.city_embed, actions_logits)

        # expanded_city_embed = self.city_embed.expand(select.size(1), -1, -1)
        # expanded_select = select.unsqueeze(-1).expand(-1,-1,128)
        # select_city_embed = torch.gather(expanded_city_embed,1, expanded_select)
        # reselect = self.action_reselector(agent_embed, select_city_embed)
        return actions_logits, agent_embed


class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        self.actions_model = ActionsModel(config)
        self.conflict_model = ConflictModel(config)
        initialize_weights(self)
        self.step = 0
        self.cfg = config

    def init_city(self, city, n_agents):
        self.actions_model.init_city(city, n_agents)
        self.step = 0

    def forward(self, agent, mask, info=None):

        if self.step == 0:
            self.actions_model.agent_decoder.init_rnn_state(agent.size(0), agent.size(1), agent.device)

        mode = "greedy" if info is None else info.get("mode", "greedy")
        use_conflict_model = True if info is None else info.get("use_conflict_model", True)
        actions_logits, agents_embed = self.actions_model(agent, mask, info)
        acts = None
        if mode == "greedy":
            # 1. 获取初始选择 --------------------------------------------------------
            if self.step == 0:
                acts_p = nn.functional.softmax(actions_logits, dim=-1)
                _, acts = acts_p[:, 0, :].topk(agent.size(1), dim=-1)
            else:
                acts = actions_logits.argmax(dim=-1)
        elif mode == "sample":
            # if self.step == 0:
            #     acts_p = nn.functional.softmax(actions_logits, dim=-1)
            #     _, acts  = acts_p[:,0,:].topk(agent.size(1), dim=-1)
            # else:
            acts = torch.distributions.Categorical(logits=actions_logits).sample()

        else:
            raise NotImplementedError
        if use_conflict_model:
            agents_logits = self.conflict_model(agents_embed, self.actions_model.city_embed, acts, info)

            agents = agents_logits.argmax(dim=-1)

            # pos = torch.arange(agents_embed.size(1), device=agents.device).unsqueeze(0).expand(agent.size(0), -1)
            pos = torch.arange(agents_embed.size(1), device=agents.device).unsqueeze(0)
            masks = torch.logical_or(agents == pos, acts == 0)
            del pos
            acts_no_conflict = torch.where(masks, acts, -1)
        else:
            agents_logits = None
            acts_no_conflict = acts
            masks = None
        self.step += 1
        a = torch.all((acts_no_conflict == -1), dim=-1)
        if torch.any(a):
            pass
        return actions_logits, agents_logits, acts, acts_no_conflict, masks


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
    from algorithm.DNN4.AgentBase import AgentBase

    agent = AgentBase(args, Config, Model)
    agent.reset_graph(graph_t)

    states_nb, actions_nb, returns_nb, masks_nb, done_nb = (
        agent._run_episode(env, graph[np.newaxis,], anum, eval_mode=False, exploit_mode="greedy"))

    from utils.GraphPlot import GraphPlot as GP

    gp = GP()
    # gp.draw_route(graph, EndInfo["trajectories"], title="random", one_first=True)
