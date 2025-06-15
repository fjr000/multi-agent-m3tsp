import torch
import torch.nn as nn
import math

from model.AttnModel.Net import MultiHeadAttention
from model.AttnModel.Net import CrossAttentionLayer, CrossAttentionCacheKVLayer
from model.AttnModel.Net import SkipConnection
from model.AttnModel.Net import SingleHeadAttention

from model.AttnModel.Utils import initialize_weights

class Config(object):
    embed_dim = 128
    dropout = 0

    city_encoder_hidden_dim = 128
    city_encoder_num_layers = 3
    city_encoder_num_heads = 8

    action_decoder_hidden_dim = 128
    action_decoder_num_layers = 2
    action_decoder_num_heads = 8

    conflict_deal_hidden_dim = 128
    conflict_deal_num_layers = 1
    conflict_deal_num_heads = 8

    action_hidden_dim = 128
    action_num_layers = 1
    action_num_heads = 8


class CityEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
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

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=10000):
        super().__init__()
        self.d_model = d_model

        self.encoding = torch.zeros(max_seq_len, d_model)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, dtype=torch.float32)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, seq_len):
        return self.encoding[:seq_len, :]


class CityEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2):
        super(CityEncoder, self).__init__()
        self.city_embedder = CityEmbedding(input_dim, embed_dim)
        self.city_self_att = nn.ModuleList(
            [
                CrossAttentionCacheKVLayer(embed_dim,num_heads=num_heads,hidden_size=hidden_dim, batch_norm=True)
                for _ in range(num_layers)
            ]
        )

        self.position_encoder = PositionalEncoder(embed_dim)
        # self.position_encoder = VectorizedRadialEncoder(embed_dim)
        self.pos_embed_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.city_embed_mean = None
        self.city_embed = None

    def update_city_mean(self, n_agent, city_mask=None, batch_mask=None):
        if city_mask is None:
            return self.city_embed_mean
        else:
            # 获取原始城市嵌入（排除最后n_agent个）
            if batch_mask is not None:
                ori_city_embed = self.city_embed[batch_mask][:, :-n_agent]
            else:
                ori_city_embed = self.city_embed[:, :-n_agent]
            # 反转掩码（True表示有效位置）
            valid_mask = ~city_mask
            valid_mask[:, 0] = True

            # 计算有效位置的数量 [B]
            valid_counts = valid_mask.sum(dim=1).clamp(min=1)  # 确保至少为1

            # 计算掩码后的均值
            masked_sum = (ori_city_embed * valid_mask.unsqueeze(-1).float()).sum(dim=1, keepdim=True)
            self.city_embed_mean = masked_sum / valid_counts.view(-1, 1, 1)
            del masked_sum, valid_counts, ori_city_embed
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
        pos_embed = self.alpha * self.pos_embed_proj(pos_embed) / n_agents
        depot_pos_embed = depot_embed + pos_embed[None, :]

        graph_embed = torch.cat([depot_embed, city_embed, depot_pos_embed], dim=1)

        for model in self.city_self_att:
            model.cache_keys(graph_embed)
            graph_embed = model(graph_embed, attn_mask=city_mask)

        self.city_embed = graph_embed

        del graph_embed, depot_pos_embed,pos_embed

        return self.city_embed  # (B,A+N,E)


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

        # 城市位置相关嵌入
        self.cur_pos_embed = nn.Linear(embed_dim, embed_dim, bias=False)

        # 距离和成本特征嵌入
        self.distance_cost_embed = nn.Sequential(
            nn.Linear(7, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU()
        )

        # 全局特征嵌入
        self.global_embed = nn.Sequential(
            nn.Linear(4, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU()
        )

        # 销售员特征融合
        self.salesman_embed = nn.Linear(embed_dim, embed_dim)

        # 图嵌入投影
        self.graph_embed = nn.Linear(embed_dim, embed_dim, bias=False)

        # depot 特征预处理
        self.depot_graph_pre_embed = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # 上下文融合模块
        self.context_fusion = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        self.global_depot_embed = None

    def cache_embed(self, global_depot_embed, n_depot_embed):
        graph_pre_embed = torch.cat([global_depot_embed.expand_as(n_depot_embed), n_depot_embed], dim=-1)
        self.global_depot_embed = self.depot_graph_pre_embed(graph_pre_embed)

    def forward(self, cities_embed, graph_embed, agent_state, batch_mask = None):
        """
        :param graph_embed
        :param agent_state: [B,M,14]
        :return:
        """
        cur_pos = cities_embed[
                  torch.arange(agent_state.size(0), device=cities_embed.device)[:, None],
                  agent_state[:, :, 1].long(),
                  :
                  ]
        cur_pos_embed = self.cur_pos_embed(cur_pos)
        distance_cost_embed = self.distance_cost_embed(agent_state[:, :, 2:9])
        global_embed = self.global_embed(agent_state[:, :, 9:])

        salesman_embed = self.salesman_embed(torch.cat([distance_cost_embed, global_embed], dim=-1))

        global_graph_embed = self.graph_embed(graph_embed).expand(-1, agent_state.size(1), -1)

        agent_embed = self.context_fusion(
            torch.cat(
                [
                    self.global_depot_embed[batch_mask] if batch_mask is not None else self.global_depot_embed,
                    cur_pos_embed,
                    salesman_embed,
                    global_graph_embed
                ], dim=-1)
        )

        del cur_pos_embed, distance_cost_embed, global_embed, salesman_embed, global_graph_embed

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
            hidden_size=hidden_dim
        )

    def cache_keys(self, city_embed):
        self.cross_att.cache_keys(city_embed)

    def forward(self, Q, cross_attn_mask=None, batch_mask=None):
        embed = self.self_att(Q)
        embed = self.cross_att(embed, cross_attn_mask, batch_mask)
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
        self.action = SingleHeadAttention(embed_dim, tanh_clip=10, use_cache=True)
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

    # def init_rnn_state(self, batch_size, agent_num, device):
    #     if isinstance(self.rnn, nn.GRU):
    #         self.rnn_state = torch.zeros(
    #             (1, batch_size * agent_num, self.rnn.hidden_size),
    #             dtype=torch.float32,
    #             device=device
    #         )
    #     elif isinstance(self.rnn, nn.LSTM):
    #         self.rnn_state = [torch.zeros(
    #             (1, batch_size * agent_num, self.rnn.hidden_size),
    #             dtype=torch.float32,
    #             device=device
    #         ),
    #             torch.zeros(
    #                 (1, batch_size * agent_num, self.rnn.hidden_size),
    #                 dtype=torch.float32,
    #                 device=device
    #             )
    #         ]
    #     elif self.rnn is None:
    #         pass
    #     else:
    #         raise NotImplementedError

    def cache_keys(self, city_embed, n_agent):
        for model in self.blocks:
            model.cache_keys(city_embed)
        self.action.cache_keys(city_embed[:, :-n_agent])
        self.agent_embed_proj.cache_embed(city_embed[:,0:1],city_embed[:, -n_agent:, :])

    def forward(self, agent, city_embed_mean, city_embed, masks, batch_mask=None):
        n_agents = agent.size(1)

        agent_embed = self.agent_embed_proj(
            city_embed[:, :-n_agents, :],
            city_embed_mean, agent,
            batch_mask = batch_mask
        )

        extra_masks = ~torch.eye(
            n_agents,
            device=agent_embed.device,
            dtype=torch.bool
        ).unsqueeze(0).expand(agent_embed.size(0), -1, -1)

        expand_masks = torch.cat([masks, extra_masks], dim=-1)

        for block in self.blocks[:-1]:
            agent_embed = block(agent_embed, expand_masks, batch_mask)

        if self.rnn is not None:
            agent_embed_shape = agent_embed.shape
            agent_embed_reshape = agent_embed.reshape(-1, 1, agent_embed.size(-1))
            agent_embed_reshape, self.rnn_state = self.rnn(agent_embed_reshape, self.rnn_state)
            agent_embed = agent_embed_reshape.reshape(*agent_embed_shape)

        agent_embed = self.blocks[-1](agent_embed, expand_masks, batch_mask)

        self.agent_embed = agent_embed

        action_logits = self.action(
            agent_embed,
            mask= masks,
            batch_mask=batch_mask
        )

        del masks, expand_masks, extra_masks

        return action_logits, self.agent_embed


class ConflictModel(nn.Module):
    def __init__(self, config: Config):
        super(ConflictModel, self).__init__()
        # 不能使用dropout
        self.city_agent_att = nn.ModuleList([
            CrossAttentionLayer(config.embed_dim, config.conflict_deal_num_heads,
                                hidden_size=config.conflict_deal_hidden_dim)
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
        self.agent_embed = None

    def init_city(self, city, n_agents, repeat_times = 1):
        """
        :param city: [B,N,2]
        :return: None
        """
        self.city = city
        self.city_embed = self.city_encoder(city, n_agents)
        if repeat_times > 1:
            self.city_embed = self.city_embed.unsqueeze(0).expand(repeat_times, -1, -1, -1).reshape(-1, self.city_embed.size(1), self.city_embed.size(2))
            self.city_encoder.city_embed = self.city_embed
        self.agent_decoder.cache_keys(self.city_embed, n_agents)

    def update_city_mean(self, n_agent, mask=None, batch_mask=None):
        self.city_embed_mean = self.city_encoder.update_city_mean(n_agent, mask, batch_mask)

    def forward(self, agent, mask, info=None):

        batch_mask = None if info is None else info.get("batch_mask", None)
        city_embed = self.city_embed
        if batch_mask is not None:
            city_embed = self.city_embed[batch_mask]
        actions_logits, agent_embed = self.agent_decoder(agent, self.city_embed_mean, city_embed, mask, batch_mask)

        del mask

        return actions_logits, agent_embed



class ConflictDeal:
    def __call__(self, *args, **kwargs):
        agent_embed, city_embed, acts = args

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

        k = agent_embed
        q = cac
        score = torch.bmm(q, k.transpose(-2, -1).contiguous())
        score = score.masked_fill(conflict_matrix, -torch.inf)

        del conflict_matrix, identity_matrix, acts_exp1, acts_exp2
        return score

class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        self.actions_model = ActionsModel(config)
        self.conflict_deal = ConflictDeal()
        initialize_weights(self)
        self.step = 0
        self.cfg = config

    def init_city(self, city, n_agents,repeat_times = 1):
        self.actions_model.init_city(city, n_agents, repeat_times)
        self.step = 0

    def forward(self, agent, mask, info=None, eval=False):
        info.update({
            "dones": None
        })

        # if self.step == 0:
        #     self.actions_model.agent_decoder.init_rnn_state(agent.size(0), agent.size(1), agent.device)

        dones = None if info is None else info.get("dones", None)
        batch_mask = None if dones is None else ~dones
        city_mask = None if info is None else info.get("mask", None)
        info.update({
            "batch_mask": batch_mask
        })

        if batch_mask is not None:
            agent = agent[batch_mask]
            city_mask = city_mask[batch_mask] if city_mask is not None else city_mask
            mask = mask[batch_mask] if mask is not None else mask

        self.actions_model.update_city_mean(agent.size(1), city_mask, batch_mask)

        mode = "greedy" if info is None else info.get("mode", "greedy")
        actions_logits, _ = self.actions_model(agent, mask, info)
        acts = None
        if mode == "greedy":

            acts = actions_logits.argmax(dim=-1)
        elif mode == "sample":
            probs = torch.softmax(actions_logits.view(-1, actions_logits.size(-1)), dim=-1)
            acts = torch.multinomial(probs, num_samples=1).squeeze(-1)
            acts = acts.view(actions_logits.size(0), actions_logits.size(1))
            # acts = torch.distributions.Categorical(logits=actions_logits).sample()

        else:
            raise NotImplementedError

        use_conflict_model = False if info is None else info.get("use_conflict_model", False)
        acts_no_conflict = None
        if use_conflict_model:
            act2agent_logits = self.conflict_deal(self.actions_model.agent_decoder.action.glimpse_Q,
                                                  self.actions_model.agent_decoder.action.glimpse_K.transpose(-2,
                                                                                                              -1).contiguous(),
                                                  acts)
            agents = act2agent_logits.argmax(dim=-1)
            pos = torch.arange(actions_logits.size(1), device=agents.device).unsqueeze(0)
            masks = torch.logical_or(agents == pos, acts == 0)
            acts_no_conflict = torch.where(masks, acts, -1)
            del pos, agents, masks

        self.step += 1

        if batch_mask is not None:
            B = batch_mask.size(0)
            A = actions_logits.size(1)
            N = actions_logits.size(2)

            final_acts = torch.zeros((B, A), dtype=torch.int64, device=actions_logits.device)
            final_acts[batch_mask] = acts
            final_acts_no_conflict = torch.zeros((B, A), dtype=torch.int64, device=actions_logits.device)
            final_acts_no_conflict[batch_mask] = acts_no_conflict

            if eval:
                return None, final_acts, final_acts_no_conflict

            final_actions_logits = torch.full((B, A, N),
                                              fill_value=-torch.inf,
                                              dtype=torch.float32,
                                              device=actions_logits.device)
            final_actions_logits[:, :, 0] = 1.0  # 模拟选择仓库
            final_actions_logits[batch_mask] = actions_logits

            actions_logits = final_actions_logits
            acts = final_acts
        return actions_logits, acts, acts_no_conflict
