import argparse

import numpy as np
from torch import inference_mode
from model.Base.Net import MultiHeadAttentionLayer, SingleHeadAttention, CrossAttentionLayer
from model.nModel.model_v1 import CityEncoder
from model.Base.Net import CrossAttentionLayer, SingleHeadAttention
import torch
import torch.nn as nn
from model.n4Model.config import Config
from model.Base.Net import initialize_weights

class AgentEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(AgentEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.depot_pos_embed = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.distance_cost_embed = nn.Linear(4, self.embed_dim)
        self.next_cost_embed = nn.Linear(4, self.embed_dim)
        self.problem_scale_embed = nn.Linear(3, self.embed_dim)
        self.graph_embed = nn.Linear(self.embed_dim, self.embed_dim)

        self.agent_embed = nn.Linear(2 * self.embed_dim, self.embed_dim)

    def forward(self,cities_embed, graph_embed, agent_state):
        """
        :param graph_embed
        :param agent_state: [B,M,14]
        :return:
        """


        # cities_expand = cities_embed.expand(agent_state.size(0), -1, -1)
        depot_pos = cities_embed[torch.arange(agent_state.size(0))[:, None, None], agent_state[:,:,:2].long(),:].reshape(agent_state.size(0),agent_state.size(1), 2*self.embed_dim)
        depot_pos_embed = self.depot_pos_embed(depot_pos)
        distance_cost_embed = self.distance_cost_embed(agent_state[:,:,2:6])
        next_cost_embed = self.next_cost_embed(agent_state[:,:,6:10])
        problem_scale_embed = self.problem_scale_embed(agent_state[:,:,10:])
        global_graph_embed = self.graph_embed(graph_embed).expand_as(depot_pos_embed)

        # co_embed = self.co_embed(agent_state[:,:,9:10])
        # agent_embed = global_graph_embed + depot_pos_embed + distance_cost_embed + problem_scale_embed
        context = torch.cat([global_graph_embed, depot_pos_embed + distance_cost_embed + next_cost_embed + problem_scale_embed], dim=-1)
        agent_embed = self.agent_embed(context)
        return agent_embed

class AgentEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2, dropout=0):
        super(AgentEncoder, self).__init__()
        self.agent_embed = AgentEmbedding(input_dim, hidden_dim, embed_dim)
        self.agent_self_att = nn.Sequential(
            *[
                MultiHeadAttentionLayer(num_heads, embed_dim, hidden_dim, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self,cities_embed, graph, agent):
        """
        :param agent: [B,N,2]
        :return:
        """
        agent_embed = self.agent_embed(cities_embed, graph, agent)
        agent_embed = self.agent_self_att(agent_embed)
        return agent_embed

class ActionDecoder(nn.Module):
    def __init__(self, hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2,dropout=0):
        super(ActionDecoder, self).__init__()
        self.agent_city_att = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads, use_FFN=True, hidden_size=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        # self.linear_forward = nn.Linear(embed_dim, embed_dim)
        self.action = SingleHeadAttention(embed_dim)
        self.num_heads = num_heads

    def forward(self, agent_embed, city_embed, masks):
        # expand_city_embed = city_embed.expand(agent_embed.size(0), -1, -1)
        expand_masks = masks.unsqueeze(1).expand(agent_embed.size(0), self.num_heads, -1, -1).reshape(agent_embed.size(0) * self.num_heads, masks.size(-2), masks.size(-1))
        # expand_masks = expand_masks.reshape(agent_embed.size(0) * self.num_heads, expand_masks.size(2), expand_masks.size(3))
        aca = agent_embed
        for model in self.agent_city_att:
            aca = model(aca, city_embed, city_embed, expand_masks)
        # cross_out = self.linear_forward(aca)
        action_logits = self.action(aca, city_embed, masks)
        return action_logits

class ConflictModel(nn.Module):
    def __init__(self, config: Config):
        super(ConflictModel, self).__init__()
        self.city_agent_att = nn.ModuleList([
            CrossAttentionLayer(config.embed_dim, config.conflict_deal_num_heads,
                                use_FFN=True, hidden_size=config.conflict_deal_hidden_dim,
                                dropout=config.dropout)
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
        identity_matrix = torch.eye(A, device=acts.device).unsqueeze(0)  # [1, A, A]
        conflict_matrix = torch.where(acts_exp1 == 0, identity_matrix, conflict_matrix)
        expand_conflict_mask = conflict_matrix.unsqueeze(1).expand(B, self.num_heads, A, A).reshape(B*self.num_heads, A, A)

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
        for att in  self.city_agent_att:
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

        # expand_graph = self.city_embed_mean.unsqueeze(1).expand(agent.size(0), agent.size(1), -1)
        # expand_graph = self.city_embed_mean.unsqueeze(1)
        agent_embed = self.agent_encoder(self.city_embed, self.city_embed_mean.unsqueeze(1), agent)

        actions_logits = self.agent_decoder( agent_embed, self.city_embed, mask)

        # agents_logits = self.deal_conflict(agent_embed, self.city_embed, actions_logits)

        # expanded_city_embed = self.city_embed.expand(select.size(1), -1, -1)
        # expanded_select = select.unsqueeze(-1).expand(-1,-1,128)
        # select_city_embed = torch.gather(expanded_city_embed,1, expanded_select)
        # reselect = self.action_reselector(agent_embed, select_city_embed)
        return actions_logits, agent_embed

class CriticModel(nn.Module):
    def __init__(self, config: Config):
        super(CriticModel, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(config.embed_dim,config.action_decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.action_decoder_hidden_dim,1),
        )

    def forward(self, agent_embed):
        value = self.critic(agent_embed)

        return value
class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        self.actions_model = ActionsModel(config)
        self.conflict_model = ConflictModel(config)
        self.critic_model = CriticModel(config)
        initialize_weights(self)
        self.step = 0
        self.cfg = config


    def init_city(self, city):
        self.actions_model.init_city(city)
        self.step = 0

    def forward(self, agent, mask, info = None):
        mode = "greedy" if info is None else info.get("mode", "greedy")
        use_conflict_model = True if info is None else info.get("use_conflict_model", True)
        actions_logits, agents_embed = self.actions_model(agent, mask, info)
        acts = None
        if mode == "greedy":
            # 1. 获取初始选择 --------------------------------------------------------
            acts = actions_logits.argmax(dim=-1)
            # if self.step == 0:
            # acts_p = nn.functional.softmax(actions_logits, dim=-1)
            #     _, acts  = acts_p[:,0,:].topk(agent.size(1), dim=-1)
        elif mode == "sample":
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

        value = self.critic_model(agents_embed)

        self.step += 1
        return actions_logits, agents_logits, acts, acts_no_conflict, masks


if __name__ == "__main__":
    from envs.MTSP.MTSP4 import MTSPEnv
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
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--grad_max_norm", type=float, default=1.0)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=2e-2)
    parser.add_argument("--batch_size", type=float, default=16)
    parser.add_argument("--city_nums", type=int, default=50)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=100000)
    parser.add_argument("--env_masks_mode", type=int, default=0,
                        help="0 for only the min cost  not allow back depot; 1 for only the max cost allow back depot")
    parser.add_argument("--eval_interval", type=int, default=100, help="eval  interval")
    parser.add_argument("--use_conflict_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--only_one_instance", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--save_model_interval", type=int, default=10000, help="save model interval")
    args = parser.parse_args()

    env = MTSPEnv({"env_masks_mode":args.env_masks_mode})

    states, info = env.reset()
    anum = info["salesmen"]
    cnum = info["cities"]
    graph = info["graph"]
    global_mask = info["mask"]
    agents_mask = info["mask"][np.newaxis].repeat(anum, axis=0)
    from algorithm.DNN_C.AgentV1 import AgentV1 as Agent
    from model.Model_AC.config import Config
    agent = Agent(args, Config)

    from envs.GraphGenerator import GraphGenerator as GG
    fig = None
    graphG = GG(args.batch_size, args.city_nums, 2)
    graph = graphG.generate()
    agent_num = args.agent_num

    act_logp, agents_logp, act_ent, agt_ent, costs = agent.run_batch_episode(env, graph, agent_num, eval_mode=False,
                                                                             info={
                                                                                 "use_conflict_model": args.use_conflict_model})
    act_loss, agents_loss, act_ent_loss, agt_ent_loss = agent.learn(act_logp, agents_logp, act_ent, agt_ent, costs)
