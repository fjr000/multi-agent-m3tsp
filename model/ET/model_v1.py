from model.Base.Net import MultiHeadAttentionLayer
from model.Base.Net import CrossAttentionLayer, SingleHeadAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Base.Net import initialize_weights
from model.Common.PositionEncoder import PositionalEncoder
import numpy as np
import math
from model.ET.config import Config

class MultiHeadAttentionCacheKV(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(MultiHeadAttentionCacheKV, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.head_dim = d_model // n_head

        self.W_kv = nn.Linear(self.d_model, self.d_model * 2, bias=False)
        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.glimpse_K = None
        self.glimpse_V = None

    def init(self, embed):
        B, A, E = embed.shape
        self.glimpse_K, self.glimpse_V = self.W_kv(embed).chunk(2, dim=-1)
        self.glimpse_K = self.glimpse_K.view(B, -1, self.n_head, self.head_dim).permute(0, 2, 3, 1)
        self.glimpse_V = self.glimpse_V.view(B, -1, self.n_head, self.head_dim).transpose(1, 2)

    def forward(self, q_embed, mask=None, batch_mask = None):
        B = q_embed.size(0)
        glimpse_Q = self.W_q(q_embed).view(B, -1, self.n_head, self.head_dim).transpose(1, 2)

        glimpse_K = self.glimpse_K
        glimpse_V = self.glimpse_V
        if batch_mask is not None:
            glimpse_K = glimpse_K[batch_mask]
            glimpse_V = glimpse_V[batch_mask]

        score = torch.matmul(glimpse_Q, glimpse_K) / math.sqrt(self.head_dim)

        if mask is not None:
            score = score.masked_fill(mask.unsqueeze(1), -torch.inf)

        attn_weight = F.softmax(score, dim=-1)
        attn_weight = self.dropout(attn_weight)

        context = torch.matmul(attn_weight, glimpse_V)
        context = context.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        out = self.out(context)
        return out


class SingleHeadAttentionCacheK(nn.Module):
    def __init__(self, d_model, tanh_clip=10, use_query_proj=False):
        super(SingleHeadAttentionCacheK, self).__init__()
        self.d_model = d_model
        self.tanh_clip = tanh_clip

        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.use_query_proj = use_query_proj
        if use_query_proj:
            self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)

        self.glimpse_K = None

    def init(self, embed):
        self.glimpse_K = self.W_k(embed)
        self.glimpse_K = self.glimpse_K.transpose(-2, -1)

    def forward(self, q_embed, mask=None, batch_mask = None):
        B = q_embed.size(0)
        glimpse_Q = q_embed
        if self.use_query_proj:
            glimpse_Q = self.W_q(q_embed)

        glimpse_K = self.glimpse_K
        if batch_mask is not None:
            glimpse_K = glimpse_K[batch_mask]

        logits = torch.matmul(glimpse_Q, glimpse_K) / math.sqrt(self.d_model)

        if self.tanh_clip > 0:
            logits = torch.tanh(logits) * self.tanh_clip

        if mask is not None:
            logits = logits.masked_fill(mask, -torch.inf)

        return logits


class CityEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(CityEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depot_embed = nn.Linear(self.input_dim, self.embed_dim)
        self.city_embed = nn.Linear(self.input_dim, self.embed_dim)
        self.position_encoder = PositionalEncoder(embed_dim)
        self.pos_embed_proj = nn.Linear(embed_dim, embed_dim)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, city, n_agents):
        """
        :param city: [B,N,2]
        :return:
        """
        depot_embed = self.depot_embed(city[:, 0:1])
        city_embed = self.city_embed(city[:, 1:])

        pos_embed = self.position_encoder(n_agents + 1).to(depot_embed.device)
        depot_embed_repeat = depot_embed.expand(-1, n_agents + 1, -1)
        pos_embed = self.alpha * self.pos_embed_proj(pos_embed) / n_agents
        depot_pos_embed = depot_embed_repeat + pos_embed[None, :]

        graph_embed = torch.cat([depot_pos_embed, city_embed], dim=1)

        return graph_embed


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
        self.position_encoder = PositionalEncoder(embed_dim)
        self.pos_embed_proj = nn.Linear(embed_dim, embed_dim)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, city, n_agents, city_mask=None):
        """
        :param city: [B,N,2]
        :return:
        """
        assert city_mask is None, "not support city_mask"

        city_embed = self.city_embed(city, n_agents)

        for model in self.city_self_att:
            city_embed = model(city_embed, key_padding_mask=city_mask)

        return (
            city_embed,  # (B,A+N,E)
            city_embed.mean(keepdim=True, dim=1)  # (batch_size, 1, embed_dim)
        )


class CityDecoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2):
        super(CityDecoder, self).__init__()

        self.project_step_context = nn.Linear(2 * embed_dim + 2, embed_dim, bias=False)
        self.dis_embed = nn.Linear(3, embed_dim, bias=False)
        self.cross_att = MultiHeadAttentionCacheKV(embed_dim, num_heads, 0.1)
        self.out = SingleHeadAttentionCacheK(embed_dim, 10, False)

    def init(self, city_embed):
        self.cross_att.init(city_embed)
        self.out.init(city_embed)

    def forward(self, state, graph_embed, city_embed):
        batch_indices = torch.arange(graph_embed.size(0), device=graph_embed.device)[:, None]
        depot_embed = city_embed[batch_indices, state.agent_id + 1, :]
        cur_pos = city_embed[batch_indices, state.cur_pos, :]

        remain_city_ratio = state.remain_city_ratio[..., None]
        remain_agent_ratio = state.remain_agent_ratio[..., None]

        step_context = self.project_step_context(
            torch.cat([depot_embed, cur_pos, remain_agent_ratio, remain_city_ratio], dim=-1))
        dis_embed = self.dis_embed(
            torch.cat([state.costs[batch_indices, state.agent_id], state.max_distance, state.remain_max_distance],
                      dim=-1))[:, None, :]
        q = step_context + graph_embed + dis_embed

        q = self.cross_att(q, state.mask[:, None, :])
        logits = self.out(q, state.mask[:, None, :])
        return logits


class Model(nn.Module):
    def __init__(self, config : Config()):
        super(Model, self).__init__()
        # encoder
        self.city_encoder = CityEncoder(2, config.city_encoder_n_hidden, config.embed_dim, config.city_encoder_n_head, config.city_encoder_n_layers)
        self.city_embed = None
        self.city_embed_mean = None
        self.graph_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        # decoder
        self.city_decoder = CityDecoder(2, config.city_encoder_n_hidden, config.embed_dim, config.city_encoder_n_head, 1)

    def init_city(self, city, n_agents):
        self.city_embed, self.city_embed_mean = self.city_encoder(city, n_agents)
        self.city_embed_mean = self.graph_proj(self.city_embed_mean)
        self.city_decoder.init(self.city_embed)

    def forward(self, state, mode = "greedy"):
        logits = self.city_decoder(state, self.city_embed_mean, self.city_embed)
        act = None
        if mode == "greedy":
            act = torch.argmax(logits, dim=-1)
        elif mode == "sample":
            dist = torch.distributions.Categorical(logits=logits)
            act = dist.sample()
        return logits, None, act, act, None


if __name__ == '__main__':
    device = torch.device('cuda:0')

    city = CityEncoder(2, 256, 128, 8, 2).to(device)
    n_agent = 3
    n_city = 10
    n_graph = 4

    from envs.DMTSP.SeqMTSP import SeqMTSPEnv

    env = SeqMTSPEnv()
    graph = env.generate_graph(n_graph, n_city)
    state = env.reset(graph, n_agent)
    state_t = state.n2t(device)

    output = city(state_t.graph, n_agent)
    pass

    MHA = MultiHeadAttentionCacheKV(128, 8, 0.1).to(device)
    MHA.init(output[0])
    mask = torch.randint(0, 2, size=(n_graph, n_agent, n_city + n_agent)).to(device).bool()
    o = MHA(output[0][:, :3], mask)

    SHA = SingleHeadAttentionCacheK(128, 10, use_query_proj=True).to(device)
    SHA.init(output[0])
    logits = SHA(o, mask)

    model = Model(input_dim=2, hidden_dim=256, embed_dim=128, num_heads=8, num_layers=3).to(device)
    model.init_city(state_t.graph, n_agent)

    done = False
    while not done:
        state_t = state.n2t(device)
        logits, _, act, _, _ = model(state_t)
        action = act.cpu().detach().numpy()
        state, r, done, info = env.step(action)
        pass
    pass
