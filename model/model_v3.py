import numpy as np
import torch
import torch.nn as nn

import model.Base.Net as Net
# from model.Base.Net import MultiHeadAttentionLayer, SingleHeadAttention, CrossAttentionLayer

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def make_fc_layer(in_features: int, out_features: int, use_bias=True, layer_norm=False):
    """Wrapper function to create and initialize a linear layer
    Args:
        in_features (int): ``in_features``
        out_features (int): ``out_features``

    Returns:
        nn.Linear: the initialized linear layer
    """

    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)

    nn.init.orthogonal(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)
    if layer_norm:
        seq = nn.Sequential(
            nn.LayerNorm(in_features),
            fc_layer
        )
        return seq
    return fc_layer


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module
        self.alpha = nn.Parameter(torch.Tensor([0.5]))

    def forward(self, input):
        module_out = self.module(input)
        return input + module_out * self.alpha


class MLP(nn.Module):
    def __init__(self, layers_dim, nonlinear_layer_last=True):
        super(MLP, self).__init__()
        self.seq = nn.Sequential()
        for i, layer in enumerate(layers_dim):
            if i < len(layers_dim) - 1:
                self.seq.append(make_fc_layer(layer, layers_dim[i + 1]))
                if i < len(layers_dim) - 2 or (nonlinear_layer_last and i == len(layers_dim) - 2):
                    self.seq.append(nn.ReLU())

    def forward(self, x):
        return self.seq(x)


class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(EmbeddingNet, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.embedding = MLP([input_dim, 256, embedding_dim], nonlinear_layer_last=False)

    def forward(self, x):
        return self.embedding(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads, dropout = 0):
        super(MultiHeadAttention, self).__init__()
        self.layer = nn.MultiheadAttention(embedding_dim, n_heads, dropout = dropout, batch_first = True)

    def forward(self, x):
        out,_ = self.layer(x, x, x)
        return out

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_heads, embedding_dim, hidden_dim, normalization='batch'):
        super(MultiHeadAttentionLayer, self).__init__()
        normalization = []
        if normalization == 'batch':
            normalization = [nn.BatchNorm1d(embedding_dim)]
        elif normalization == 'layer':
            normalization = [nn.LayerNorm(embedding_dim)]
        else:
            normalization = []
        self.layers = nn.Sequential(
            SkipConnection(
                MultiHeadAttention(
                    embedding_dim,
                    n_heads,
                    dropout=0
                )
            ),

            *normalization,

            SkipConnection(
                nn.Sequential(
                    nn.Linear(embedding_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, embedding_dim)
                )
            ),

            *normalization,
        )

    def forward(self, x):
        return self.layers(x)


class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim
        self.key_dim = self.value_dim
        self.tanh_clipping = 10
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
                :param q: queries (batch_size, n_query, input_dim)
                :param h: data (batch_size, graph_size, input_dim)
                :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
                Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
                :return:
                """
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)  # n_query = target_size in tsp

        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim

        h_flat = h.reshape(-1, input_dim)  # (batch_size*graph_size)*input_dim
        q_flat = q.reshape(-1, input_dim)  # (batch_size*n_query)*input_dim

        shape_k = (batch_size, target_size, -1)
        shape_q = (batch_size, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(shape_k)  # batch_size*targets_size*key_dim

        U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))  # batch_size*n_query*targets_size
        U = self.tanh_clipping * torch.tanh(U)

        if mask is not None:
            # mask = mask.view(batch_size, 1, target_size).expand_as(U)  # copy for n_heads times
            # U = U-1e8*mask  # ??
            U[mask<1e-8] = -1e8
        # attention = torch.log_softmax(U, dim=-1)  # batch_size*n_query*targets_size

        # out = attention

        return U


class SelfAttentionLayer(nn.Module):
    # For not self attention
    def __init__(self, embedding_dim, num_heads, use_FFN=True, hidden_size=128):
        super(SelfAttentionLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.use_FFN = use_FFN
        self.multiHeadAttention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True, )
        self.normalization1 = nn.BatchNorm1d(embedding_dim)
        if self.use_FFN:
            self.feedForward = nn.Sequential(nn.Linear(embedding_dim, hidden_size),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(hidden_size, embedding_dim))
            self.normalization2 = nn.BatchNorm1d(embedding_dim)

    def forward(self, x, attn_mask=None):
        assert x.shape[-1] == self.embedding_dim, \
            f"q.shape == k.shape == v.shape  is [B,S,{self.embedding_dim}]"
        shape = x.shape
        input_ln = self.normalization1(x.view(-1, self.embedding_dim)).view(*shape)
        att_out, _ = self.multiHeadAttention(input_ln, input_ln, input_ln, attn_mask=attn_mask)
        hidden = att_out + x
        hidden_ln = self.normalization2(hidden.view(-1, self.embedding_dim)).view(*shape)
        if self.use_FFN:
            fn_out = self.feedForward(hidden_ln)
            out = fn_out + hidden
        else:
            out = hidden
        return out


class CrossAttentionLayer(nn.Module):
    # For not self attention
    def __init__(self, embedding_dim, num_heads, use_FFN=True, hidden_size=128):
        super(CrossAttentionLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.use_FFN = use_FFN
        self.multiHeadAttention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True, )
        self.normalization1 = nn.BatchNorm1d(embedding_dim)
        if self.use_FFN:
            self.feedForward = nn.Sequential(nn.Linear(embedding_dim, hidden_size),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(hidden_size, embedding_dim))
            self.normalization2 = nn.BatchNorm1d(embedding_dim)

    def forward(self, q, k, v, attn_mask=None):
        assert q.shape[-1] == self.embedding_dim and k.shape[-1] == self.embedding_dim and len(q.shape) == len(
            k.shape) and q.shape[0] == k.shape[0] and k.shape == v.shape, \
            f"q.shape == k.shape == v.shape  is [B,S,{self.embedding_dim}]"

        att_out, _ = self.multiHeadAttention(q, k, v, attn_mask=attn_mask)
        hidden = att_out + q
        shape = hidden.shape
        hidden_ln = self.normalization1(hidden.view(-1,self.embedding_dim)).view(*shape)
        if self.use_FFN:
            fn_out = self.feedForward(hidden_ln)
            out = fn_out + hidden_ln
            out = self.normalization2(out.view(-1, self.embedding_dim)).view(*shape)
        else:
            out = hidden_ln
        return out


class SelfEncoder(nn.Module):
    def __init__(self, input_dims, embedding_dim, num_heads, use_FFN=True, hidden_size=128):
        assert isinstance(input_dims, list) or isinstance(input_dims, tuple), "input_dims must be a list or tuple"
        super(SelfEncoder, self).__init__()
        self.embedding_layer_num = len(input_dims)
        self.embedding_layers = nn.ModuleList()
        self.embedding_dim = embedding_dim
        self.input_dims = input_dims
        for i in range(self.embedding_layer_num):
            self.embedding_layers.append(EmbeddingNet(input_dims[i], embedding_dim))

        self.att = SelfAttentionLayer(embedding_dim, num_heads, use_FFN, hidden_size)

    def forward(self, x, attn_mask=None):
        assert isinstance(x, list) or isinstance(x, tuple), "input must be a list or tuple"
        assert len(x) == self.embedding_layer_num, f"input shoule be split into {self.embedding_layer_num} type"
        for xx, dim in zip(x, self.input_dims):
            # assert len(xx.shape) == 3, "the shape of input data element must be [B, Seq, input_dim ]"
            assert xx.shape[-1] == dim, "the shape of input data element must be [B, Seq, input_dim ]"
            assert xx.shape[0] == x[0].shape[0], "the batch size of input data must be same"
        embeddings = []
        for i in range(self.embedding_layer_num):
            embedding = self.embedding_layers[i](x[i])
            embeddings.append(embedding)
        if len(x) > 1:
            embed = torch.cat(embeddings, dim=-2)
        else:
            embed = embeddings[0]
        out = self.att(embed, attn_mask)
        split_dim = [dd.shape[1] for dd in x]
        out = torch.split(out, split_dim, dim=-2)
        return out


class SingleDecoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, use_FFN=True, hidden_size=128):
        super(SingleDecoder, self).__init__()

        self.CA1 = CrossAttentionLayer(embedding_dim, num_heads, use_FFN, hidden_size)
        self.CA2 = CrossAttentionLayer(embedding_dim, num_heads, use_FFN, hidden_size)
        self.SAO = SingleHeadAttention(embedding_dim)

    def forward(self, agents_embeddings, cities_embeddings, attn_mask=None):
        h_c = self.CA1(agents_embeddings, cities_embeddings, cities_embeddings, attn_mask)
        h_c_embed = self.SAO(h_c, cities_embeddings, attn_mask)
        next_idx = torch.argmax(h_c_embed, dim=-2)

        return next_idx


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

        self.cur_city_embed = nn.Linear(self.input_dim, self.embed_dim)
        self.last_city_embed = nn.Linear(self.input_dim, self.embed_dim)
        self.global_graph_embed = nn.Linear(self.embed_dim, self.embed_dim)
        #
        # self.cur_city_embed = nn.Sequential(
        #     nn.Linear(self.input_dim, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, self.embed_dim),
        # )
        #
        # self.last_city_embed = nn.Sequential(
        #     nn.Linear(self.input_dim, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, self.embed_dim),
        # )

        self.agent_embed = nn.Linear(3 * self.embed_dim, self.embed_dim)

    def forward(self, graph_embed, agent_state_list):
        """
        :param graph_embed
        :param agent_state: [[B,M,2],...]
        :return:
        """
        global_graph_embed = self.global_graph_embed(graph_embed)
        last_city_embed = self.last_city_embed(agent_state_list[0])
        cur_city_embed = self.cur_city_embed(agent_state_list[1])
        city_embed = torch.cat([global_graph_embed, last_city_embed, cur_city_embed], dim=-1)
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
        self.linear_forward = nn.Linear(embed_dim, embed_dim)
        self.action = SingleHeadAttention(embed_dim)
        self.num_heads = num_heads

    def forward(self, agent_embed, city_embed, masks):
        expand_city_embed = city_embed.expand(agent_embed.size(0), -1, -1)
        expand_masks = masks.unsqueeze(1).expand(agent_embed.size(0), self.num_heads, -1, -1)
        expand_masks = expand_masks.reshape(agent_embed.size(0) * self.num_heads, expand_masks.size(2), expand_masks.size(3))
        aca = self.agent_city_att(agent_embed, expand_city_embed, expand_city_embed, expand_masks)
        cross_out = self.linear_forward(aca)
        action_logits = self.action(cross_out, expand_city_embed, masks)
        return action_logits


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
        """
        :param city: [B,N,2]
        :return: None
        """
        self.city_embed = self.city_encoder(city)

    def forward(self, agent, mask):
        graph = torch.mean(self.city_embed, dim=1)
        expand_graph = graph.unsqueeze(0).expand(agent[0].size(0), agent[0].size(1), -1)
        agent_embed = self.agent_encoder(expand_graph, agent)
        actions_logits = self.agent_decoder(agent_embed, self.city_embed, mask)

        # expanded_city_embed = self.city_embed.expand(select.size(1), -1, -1)
        # expanded_select = select.unsqueeze(-1).expand(-1,-1,128)
        # select_city_embed = torch.gather(expanded_city_embed,1, expanded_select)
        # reselect = self.action_reselector(agent_embed, select_city_embed)
        return actions_logits


if __name__ == "__main__":
    from envs.MTSP.MTSP import MTSPEnv
    from utils.TensorTools import _convert_tensor

    env = MTSPEnv()
    cfg = {
        "city_nums": (10, 10),
        "agent_nums": (3, 3),
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
    agents_last_states = info["actors_last_states"]
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
    mode = "sample"
    while not done:
        state_t = _convert_tensor(states, device="cuda", target_shape_dim=3)
        last_state_t = _convert_tensor(agents_last_states, device="cuda", target_shape_dim=3)
        agents_mask_t = _convert_tensor(agents_mask, device="cuda", target_shape_dim=3)
        actions_logits  = model([last_state_t, state_t], agents_mask_t)
        # action = np.random.choice(action_to_chose, anum, replace=False)
        dist = torch.distributions.Categorical(logits=actions_logits)
        if mode == "greedy":
            actions = torch.argmax(actions_logits, dim=-1)
            logprob = dist.log_prob(actions)
        elif mode == "sample":
            actions = dist.sample()
            logprob = dist.log_prob(actions)
        actions_numpy = actions.squeeze(0).cpu().numpy()
        states, reward, done, info = env.step(actions_numpy+1)
        global_mask = info["global_mask"]
        agents_mask = info["agents_mask"]
        agents_way = info["agents_way"]
        agents_last_states = info["actors_last_states"]
        if done:
            EndInfo.update(info)
    loss = reward
    print(f"reward:{reward}")
    print(f"trajectory:{EndInfo}")
    from utils.GraphPlot import GraphPlot as GP

    gp = GP()
    gp.draw_route(graph, EndInfo["actors_trajectory"], title="random", one_first=True)
