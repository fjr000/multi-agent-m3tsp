import argparse
import numpy as np
from model.Base.Net import MultiHeadAttentionLayer, MultiHeadAttention
from model.Base.Net import CrossAttentionLayer, SingleHeadAttention
import torch
import torch.nn as nn
from model.n4Model.config import Config
from model.Base.Net import initialize_weights
import math
from model.Base.Net import SkipConnection
from model.ET.model_v1 import MultiHeadAttentionCacheKV, SingleHeadAttentionCacheK
from model.n4Model.model_v4 import CrossAttentionCacheKVLayer


class CityEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.depot_embed = nn.Linear(input_dim, embed_dim)
        self.city_embed = nn.Linear(input_dim, embed_dim)

    def forward(self, city):
        """Embed city coordinates.
        Args:
            city: [B, N, 2] tensor of city coordinates
        Returns:
            Tuple of (depot_embed, city_embed)
        """
        return self.depot_embed(city[:, :1]), self.city_embed(city[:, 1:])

class GateConnection(nn.Module):
    def __init__(self, model, embed_dim):
        super(GateConnection, self).__init__()
        self.model = model
        self.gate = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        gate = self.gate(x)
        return x + gate * self.model(x)

from model.Common.MHA import Normalization
class BNMHAGATEFFNLayer(nn.Module):
    def __init__(self, n_head, embed_dim, hidden_dim, normalization='batch', dropout=0):
        super(BNMHAGATEFFNLayer, self).__init__()
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.attn = GateConnection(
            nn.Sequential(
                Normalization(self.embed_dim, normalization),
                MultiHeadAttention(
                    embed_dim,
                    n_head,
                    dropout=dropout
                )
            )
            ,embed_dim=self.embed_dim
        )
        self.ffn = GateConnection(
            nn.Sequential(
                Normalization(self.embed_dim, normalization),
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim)
            )
            , embed_dim=self.embed_dim
        )

    def forward(self, x, key_padding_mask = None,masks = None):
        o = self.attn(x)
        o = self.ffn(o)
        return o
class CityEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2):
        super(CityEncoder, self).__init__()
        self.city_embedder = CityEmbedding(input_dim, hidden_dim, embed_dim)
        self.city_self_att = nn.ModuleList(
            [
                BNMHAGATEFFNLayer(num_heads, embed_dim, hidden_dim, normalization='batch')
                for _ in range(num_layers)
            ]
        )

        # self.position_encoder = PositionalEncoder(embed_dim)
        self.position_encoder = VectorizedRadialEncoder(embed_dim)
        self.pos_embed_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.alpha = nn.Parameter(torch.tensor([0.1]))
        self.city_embed_mean = None
        self.city_embed = None

    def update_city_mean(self, n_agent,city_mask = None):
        if(city_mask is None):
            return self.city_embed_mean
        else:
            # 获取原始城市嵌入（排除最后n_agent个）
            ori_city_embed = self.city_embed[:, :-n_agent]

            # 反转掩码（True表示有效位置）
            valid_mask = ~city_mask
            valid_mask[:,0] = True

            # 计算有效位置的数量 [B]
            valid_counts = valid_mask.sum(dim=1).clamp(min=1)  # 确保至少为1

            # 计算掩码后的均值
            masked_sum = (ori_city_embed * valid_mask.unsqueeze(-1).float()).sum(dim=1, keepdim=True)
            mean = masked_sum / valid_counts.view(-1, 1, 1)
            self.city_embed_mean = mean
            return self.city_embed_mean

    def forward(self, city, n_agents, city_mask=None):
        """
        :param city: [B,N,2]
        :return:
        """

        if city_mask is not None:
            city_mask[:, 0] = False

        depot_embed, city_embed = self.city_embedder(city)

        pos_embed = self.position_encoder(n_agents).to(city.device)
        pos_embed = self.alpha * self.pos_embed_proj(pos_embed)
        depot_pos_embed = depot_embed + pos_embed[None, :]

        graph_embed = torch.cat([depot_embed, city_embed, depot_pos_embed], dim=1)

        for model in self.city_self_att:
            graph_embed = model(graph_embed, key_padding_mask=city_mask)

        self.city_embed_mean = graph_embed.mean(keepdim=True, dim=1)  # (batch_size, 1, embed_dim) mean(A+E)
        self.city_embed = graph_embed
        return (
            self.city_embed,  # (B,A+N,E)
            self.city_embed_mean
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
        return self.position_encoding[:seq_len, :]


class DynamicPositionalEncoder(nn.Module):
    def __init__(self, d_model):
        """
        初始化位置编码模块。
        :param d_model: 嵌入维度（必须是偶数）
        """
        super(DynamicPositionalEncoder, self).__init__()
        self.d_model = d_model

    def forward(self, num_agents):
        """
        根据智能体数量动态生成位置编码。
        :param num_agents: 当前智能体数量 (int)
        :return: 嵌入张量 (Tensor)，形状为 (num_agents, d_model)
        """
        # 动态计算频率缩放因子
        scale_factor = 10 * math.log(num_agents + 1)  # 根据智能体数量调整

        # 创建位置编码
        position_encoding = torch.zeros(num_agents, self.d_model)
        position = torch.arange(0, num_agents, dtype=torch.float).unsqueeze(1)  # [num_agents, 1]
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(scale_factor) / self.d_model))

        # 使用正弦和余弦函数生成位置编码
        position_encoding[:, 0::2] = torch.sin(position * div_term)  # 偶数索引：正弦
        position_encoding[:, 1::2] = torch.cos(position * div_term)  # 奇数索引：余弦

        return position_encoding


class VectorizedRadialEncoder(nn.Module):
    """
    高效向量化的放射状位置编码器
    为MINMAXMTSP问题提供智能体的空间分配编码
    """

    def __init__(self, d_model, base=10000):
        super(VectorizedRadialEncoder, self).__init__()
        assert d_model % 2 == 0, "模型维度必须是偶数"
        self.d_model = d_model
        self.base = base

        # 预计算频率因子以提高效率
        self.register_buffer(
            "freq_factors",
            1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        )

    def forward(self, num_agents):
        """
        批量生成所有智能体的放射状编码

        :param num_agents: 智能体数量
        :return: 位置编码 [num_agents, d_model]
        """
        # 生成均匀分布在圆周上的角度 [num_agents]
        angles = torch.linspace(0, 2 * math.pi, num_agents + 1)[:-1]
        device = self.freq_factors.device
        angles = angles.to(device)

        # 创建编码矩阵 [num_agents, d_model]
        encoding = torch.zeros(num_agents, self.d_model, device=device)

        # 向量化同时计算所有智能体的编码
        # 将角度扩展为 [num_agents, 1] 与频率因子 [d_model/2] 相乘
        angles_expanded = angles.unsqueeze(1)  # [num_agents, 1]

        # 计算所有频率的角度参数 [num_agents, d_model/2]
        args = angles_expanded * self.freq_factors

        # 一次性填充所有正弦项（偶数索引）
        encoding[:, 0::2] = torch.sin(args)

        # 一次性填充所有余弦项（奇数索引）
        encoding[:, 1::2] = torch.cos(args)

        return encoding

class AgentEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(AgentEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.depot_pos_embed = nn.Sequential(
            nn.Linear(2 * self.embed_dim, self.embed_dim),
            # nn.LayerNorm(self.embed_dim),
        )
        self.distance_cost_embed = nn.Sequential(
            nn.Linear(7, self.embed_dim),
            # nn.LayerNorm(self.embed_dim),
        )
        self.global_embed = nn.Sequential(
            nn.Linear(4, self.embed_dim),
            # nn.LayerNorm(self.embed_dim),
        )
        self.graph_embed = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            # nn.LayerNorm(self.embed_dim),
        )
        self.context = nn.Sequential(
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

    def forward(self, cities_embed, n_depot_embed, graph_embed, agent_state):
        """
        :param graph_embed
        :param agent_state: [B,M,14]
        :return:
        """
        cur_pos = cities_embed[
                  torch.arange(agent_state.size(0))[:, None],
                  agent_state[:, :, 1].long(),
                  :
                  ]
        depot_pos = torch.cat([n_depot_embed, cur_pos], dim=-1)
        depot_pos_embed = self.depot_pos_embed(depot_pos)
        distance_cost_embed = self.distance_cost_embed(agent_state[:, :, 2:9])
        global_embed = self.global_embed(agent_state[:,:,9:])
        global_graph_embed = self.graph_embed(graph_embed).expand(-1, agent_state.size(1), -1)

        agent_embed = self.context(
            torch.cat(
                [
                    depot_pos_embed,
                    distance_cost_embed,
                    global_embed,
                    global_graph_embed
                ], dim=-1)
        )
        return agent_embed


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads):
        super(DecoderBlock, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_heads = num_heads
        self.self_att = nn.Sequential(
            SkipConnection(
                MultiHeadAttention(self.embed_dim, self.n_heads),
            ),
            nn.LayerNorm(embed_dim)
        )
        self.cross_att = CrossAttentionCacheKVLayer(
            self.embed_dim,
            self.n_heads,
            use_FFN=True,
            hidden_size=hidden_dim
        )

    def init(self, city_embed):
        self.cross_att.init(city_embed)

    def forward(self, Q, cross_attn_mask=None):
        embed = self.self_att(Q)
        embed = self.cross_att(embed, cross_attn_mask)
        return embed


class ActionDecoder(nn.Module):
    def __init__(self, hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2, dropout=0, rnn_type='GRU'):
        super(ActionDecoder, self).__init__()
        self.agent_embed_proj = AgentEmbedding(embed_dim, hidden_dim, embed_dim)

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(embed_dim, hidden_dim, num_heads)
                for _ in range(num_layers)
            ]
        )

        # self.linear_forward = nn.Linear(embed_dim, embed_dim)
        self.action = SingleHeadAttentionCacheK(embed_dim)
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
        elif rnn_type is None:
            self.rnn = None
        else:
            raise NotImplementedError

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
        elif self.rnn is None:
            pass
        else:
            raise NotImplementedError

    def init(self, city_embed, n_agent):
        for model in self.blocks:
            model.init(city_embed)
        self.action.init(city_embed[:,:-n_agent])

    def forward(self, agent, city_embed_mean, city_embed, masks):
        n_agents = agent.size(1)

        agent_embed = self.agent_embed_proj(
            city_embed[:, :-n_agents, :],
            city_embed[:, -n_agents:, :],
            city_embed_mean, agent
        )

        extra_masks = ~torch.eye(
            n_agents,
            device=agent_embed.device,
            dtype=torch.bool
        ).unsqueeze(0).expand(agent_embed.size(0), -1, -1)

        expand_masks = torch.cat([masks, extra_masks], dim=-1)
        # expand_masks = expand_masks.unsqueeze(1).expand(
        #     agent_embed.size(0), self.num_heads, -1, -1
        # ).reshape(
        #     -1, *expand_masks.shape[-2:]
        # )

        for block in self.blocks[:-1]:
            agent_embed = block(agent_embed, expand_masks)

        if self.rnn is not None:
            agent_embed_shape = agent_embed.shape
            agent_embed_reshape = agent_embed.reshape(-1, 1, agent_embed.size(-1))
            agent_embed_reshape, self.rnn_state = self.rnn(agent_embed_reshape, self.rnn_state)
            agent_embed = agent_embed_reshape.reshape(*agent_embed_shape)

        agent_embed = self.blocks[-1](agent_embed, expand_masks)

        # comm = agent_embed[..., :agent_embed.size(2) // 4]
        # # comm = comm_model(comm)
        # comm, _ = comm.max(dim=1, keepdim=True)
        # comm = comm.expand(-1, agent_embed.size(1), -1)
        # priv = agent_embed[..., agent_embed.size(2) // 4:]
        # agent_embed = torch.cat([comm, priv], dim=-1)

        self.agent_embed = agent_embed

        action_logits = self.action(
            agent_embed,
            masks
        )

        del masks, expand_masks

        return action_logits, self.agent_embed


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

        self.agent_decoder = ActionDecoder(config.action_decoder_hidden_dim, config.embed_dim,
                                           config.action_decoder_num_heads, config.action_decoder_num_layers,
                                           dropout=config.dropout,
                                           rnn_type=None,
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
        self.agent_decoder.init(self.city_embed, n_agents)

    def update_city_mean(self, n_agent ,mask = None):
        self.city_embed_mean = self.city_encoder.update_city_mean(n_agent, mask)

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

        actions_logits, agent_embed = self.agent_decoder(agent, self.city_embed_mean, self.city_embed, mask)
        # x = torch.all(torch.isinf(actions_logits), dim=-1)
        # xxx = torch.isnan(actions_logits).any().item()
        # xx = x.any().item()
        # if xx or xxx:
        #     a = torch.argwhere(torch.isnan(actions_logits))
        #     pass
        # a = torch.argwhere(x)

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

    def forward(self, agent, mask, info=None, eval=False):

        if self.step == 0:
            self.actions_model.agent_decoder.init_rnn_state(agent.size(0), agent.size(1), agent.device)

        city_mask = None if info is None else info.get("mask",None)
        self.actions_model.update_city_mean(agent.size(1),city_mask)

        mode = "greedy" if info is None else info.get("mode", "greedy")
        use_conflict_model = True if info is None else info.get("use_conflict_model", True)
        actions_logits, agents_embed = self.actions_model(agent, mask, info)
        acts = None
        if mode == "greedy":
            # 1. 获取初始选择 --------------------------------------------------------
            # if self.step == 0:
            #     acts_p = nn.functional.softmax(actions_logits, dim=-1)
            #     _, acts = acts_p[:, 0, :].topk(agent.size(1), dim=-1)
            # else:
            acts = actions_logits.argmax(dim=-1)
                # acts = actions_logits.argmax(dim=-1)
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
