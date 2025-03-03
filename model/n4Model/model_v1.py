import argparse
import numpy as np
from torch import inference_mode

from model.nModel.model_v1 import CityEncoder, AgentEncoder
from model.Base.Net import CrossAttentionLayer, SingleHeadAttention
import torch
import torch.nn as nn
from model.n4Model.config import Config

class ActionDecoder(nn.Module):
    def __init__(self, hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2):
        super(ActionDecoder, self).__init__()
        self.agent_city_att = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads, use_FFN=True, hidden_size=hidden_dim)
            for _ in range(num_layers)
        ])
        # self.linear_forward = nn.Linear(embed_dim, embed_dim)
        self.action = SingleHeadAttention(embed_dim)
        self.num_heads = num_heads

    def forward(self, agent_embed, city_embed, masks):
        expand_city_embed = city_embed.expand(agent_embed.size(0), -1, -1)
        expand_masks = masks.unsqueeze(1).expand(agent_embed.size(0), self.num_heads, -1, -1)
        expand_masks = expand_masks.reshape(agent_embed.size(0) * self.num_heads, expand_masks.size(2), expand_masks.size(3))
        aca = agent_embed
        for model in self.agent_city_att:
            aca = model(aca, expand_city_embed, expand_city_embed, expand_masks)
        # cross_out = self.linear_forward(aca)
        action_logits = self.action(aca, expand_city_embed, masks)
        return action_logits

class ConflictModel(nn.Module):
    def __init__(self, config: Config):
        super(ConflictModel, self).__init__()
        self.city_agent_att = nn.ModuleList([
            CrossAttentionLayer(config.embed_dim, config.conflict_deal_num_heads,
                                use_FFN=True, hidden_size=config.conflict_deal_hidden_dim)
            for _ in range(config.conflict_deal_num_layers)
        ])
        # self.linear_forward = nn.Linear(embed_dim, embed_dim)
        self.agents = SingleHeadAttention(config.embed_dim)
        self.num_heads = config.conflict_deal_num_heads

    def forward(self, agent_embed, city_embed, acts, info = None):
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
        conflict_matrix = (acts_exp1 == acts_exp2).float()  # [B,5,5]
        expand_conflict_mask = conflict_matrix.unsqueeze(1).expand(B, self.num_heads, A, A).reshape(B*self.num_heads, A, A)

        # 3. 提取候选城市特征 -----------------------------------------------------
        selected_cities = torch.gather(
            city_embed.expand(B,-1,-1),
            1,
            acts.unsqueeze(-1).expand(-1, -1, E)
        )  # [B,5,E]

        # 4. 注意力重新分配 ------------------------------------------------------
        # Q: 候选城市特征 [B,5,E]
        # K/V: 智能体特征 [B,5,E]
        cac = selected_cities
        for att in  self.city_agent_att:
            cac = att(cac, agent_embed, agent_embed, expand_conflict_mask)

        agents_logits = self.agents(cac, agent_embed, conflict_matrix)
        #
        # agents = nn.functional.softmax(agents_logits, dim=-1).argmax(dim=-1)
        # pos = torch.arange(A, device=agents.device).unsqueeze(0).expand(B, -1)
        # masks = (agents == pos).float()

        return agents_logits


class ActionsModel(nn.Module):
    def __init__(self, config: Config):
        super(ActionsModel, self).__init__()
        self.city_encoder = CityEncoder(2, config.city_encoder_hidden_dim, config.embed_dim,
                                        config.city_encoder_num_heads, config.city_encoder_num_layers
                                        )
        self.agent_encoder = AgentEncoder(config.agent_dim, config.agent_encoder_hidden_dim, config.embed_dim,
                                          config.agent_encoder_num_heads, config.agent_encoder_num_layers
                                          )
        self.agent_decoder = ActionDecoder(config.action_decoder_hidden_dim, config.embed_dim,
                                           config.action_decoder_num_heads, config.action_decoder_num_layers
                                           )
        self.city_embed = None
        self.city_embed_mean = None

    def init_city(self, city):
        """
        :param city: [B,N,2]
        :return: None
        """
        self.city_embed = self.city_encoder(city)
        self.city_embed_mean = torch.mean(self.city_embed, dim=1)

    def forward(self, agent, mask, info = None):
        # batch_mask = mask[:,0,:].unsqueeze(-1).expand(mask.size(0),mask.size(2),self.city_embed.shape[-1])
        # ori_expand_graph = self.city_embed.expand(*batch_mask.shape)
        # mask_expand_graph = ori_expand_graph * batch_mask
        # mask_sum_expand_graph = mask_expand_graph.sum(1)
        # non_zero_count = batch_mask.sum(1)
        # avg_graph = mask_sum_expand_graph / non_zero_count

        expand_graph = self.city_embed_mean.unsqueeze(1).expand(agent.size(0), agent.size(1), -1)
        agent_embed = self.agent_encoder(expand_graph, agent)

        actions_logits = self.agent_decoder(agent_embed, self.city_embed, mask)

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

    def init_city(self, city):
        self.actions_model.init_city(city)

    def forward(self, agent, mask, info = None):
        mode = "greedy" if info is None else info.get("mode", "greedy")
        actions_logits, agents_embed = self.actions_model(agent, mask, info)
        acts = None
        if mode == "greedy":
            # 1. 获取初始选择 --------------------------------------------------------
            acts = nn.functional.softmax(actions_logits, dim=-1).argmax(dim=-1)  # [B,5] 每个智能体选的城市索引
        elif mode == "sample":
            acts = torch.distributions.Categorical(logits=actions_logits).sample()
        else:
            raise NotImplementedError

        agents_logits = self.conflict_model(agents_embed, self.actions_model.city_embed, acts, info)

        agents_dist = torch.distributions.Categorical(logits=agents_logits)
        agents = None
        if mode == "greedy":
            agents = nn.functional.softmax(agents_logits, dim=-1).argmax(dim=-1)
        elif mode == "sample":
            agents = agents_dist.sample()
        else:
            raise NotImplementedError

        pos = torch.arange(agents_embed.size(1), device=agents.device).unsqueeze(0).expand(agent.size(0), -1)
        masks = agents == pos

        acts_no_conflict = torch.where(masks, acts, -1)

        return actions_logits, agents_logits, acts, acts_no_conflict


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