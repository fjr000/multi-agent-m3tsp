import argparse

import numpy as np
from torch import inference_mode
from torch.distributions import Categorical

from model.nModel.model_v1 import CityEncoder
from model.Base.Net import CrossAttentionLayer, SingleHeadAttention
# from model.Base.Net import SkipConnection, MultiHeadAttention
import torch
import torch.nn as nn
from model.SeqModel.config import ModelConfig as Config
from model.Base.Net import initialize_weights
import math

from model.RefModel.AgentAttentionCritic import AgentAttentionCritic
from model.RefModel.MHA import MultiHeadAttentionLayer, MultiHeadAttention, Normalization
from model.RefModel.PositionEncoder import PositionalEncoder
from model.RefModel.CityAttentionEncoder import CityAttentionEncoder
from model.RefModel.AgentAttentionEncoder import AgentAttentionEncoder, AgentAttentionRNNEncoder
from model.RefModel.AgentCityAttentionDecoder import AgentCityAttentionDecoder

class Encoder(nn.Module):
    def __init__(self, config: Config):
        super(Encoder, self).__init__()
        self.config = config
        self.city_encoder = CityAttentionEncoder(config.city_encoder_config.city_encoder_num_heads,
                                                 config.city_encoder_config.embed_dim,
                                                 config.city_encoder_config.city_encoder_num_layers,
                                                 2,
                                                 'batch',
                                                 config.city_encoder_config.city_encoder_hidden_dim,)
        self.agent_encoder = AgentAttentionEncoder(input_dim=2,
                                                   hidden_dim=config.actions_model_config.agent_encoder_hidden_dim,
                                                   embed_dim=config.embed_dim,
                                                   num_heads=config.actions_model_config.agent_encoder_num_heads,
                                                   num_layers=config.actions_model_config.agent_encoder_num_layers,
                                                   norm="layer")

        self.agent_city_encoder = AgentCityAttentionDecoder(hidden_dim=config.actions_model_config.action_decoder_hidden_dim,
                                                            embed_dim=config.embed_dim,
                                                            num_heads=config.actions_model_config.action_decoder_num_heads,
                                                            num_layers=config.actions_model_config.action_decoder_num_layers,
                                                            norm="layer")
        self.value = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(config.embed_dim // 2, 1),
        )

        self.city = None
        self.city_embed = None
        self.city_embed_mean = None
        self.nodes_embed = None

    def init_city(self, city, n_agents):
        """
        :param city: [B,N,2]
        :return: None
        """
        self.city = city
        self.city_embed, self.city_embed_mean = self.city_encoder(city, n_agents)

    def forward(self, agent_states, agents_self_mask=None, agents_city_mask=None):

        n_agents = agent_states.size(1)
        agent_embed = self.agent_encoder(self.city_embed[:,:-n_agents,:], self.city_embed[:,-n_agents:,:], self.city_embed_mean, agent_states, agents_self_mask)
        agent_embed,_ = self.agent_city_encoder(agent_embed, self.city_embed, agents_city_mask)
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
            self.n_heads,
            self.embed_dim,
            self.embed_dim,
        )
        self.norm = nn.LayerNorm(self.embed_dim)

        self.cross_attn = MultiHeadAttentionLayer(
            self.embed_dim,
            self.embed_dim,
            self.hidden_dim,
            self.n_heads,
            'layer',
        )

    def forward(self, action_embed, agent_embed, attn_mask):
        kv = self.self_attn(action_embed, mask=attn_mask)
        kv = self.norm(action_embed + kv)

        embed = self.cross_attn(agent_embed, kv, attn_mask)

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

    def forward(self, action_embed, agent_embed, attn_mask, city_embed, city_mask, n_agent, idx=None):
        embed = self.act_embed(action_embed)
        # expand_mask = attn_mask[:, None].expand(-1, self.n_heads, -1, -1).reshape(-1, attn_mask.size(-2),
        #                                                                           attn_mask.size(-1))
        for model in self.decoders:
            embed = model(embed, agent_embed, attn_mask)

        if idx is not None:
            act_logits = self.act(embed[:, idx:idx + 1, :], city_embed[:,:-n_agent,:], city_mask)
        else:
            act_logits = self.act(embed, city_embed[:,:-n_agent,:], city_mask)
        return act_logits


class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.encoder = Encoder(config)
        self.decoder = ActionsDecoder(config)

    def init_city(self, city, n_agents):
        self.encoder.init_city(city, n_agents)

    def autoregressive_forward(self, agent_states, salesmen_mask=None, mode="greedy"):
        # seq
        B, A, _ = agent_states.shape

        agent_embed, V = self.encoder(agent_states, agents_city_mask=salesmen_mask)

        cur_pos = self.encoder.city_embed[torch.arange(agent_states.size(0))[:, None], agent_states[:, :, 1].long(), :]
        cur_pos_mean = cur_pos.mean(dim=1)
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
                                      agents_mask[:, :idx + 1, :idx + 1], self.encoder.city_embed, salesman_mask, A,
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
        self.init_city(batch_graph, agent_states.size(1))

        B, A, _ = agent_states.shape
        # N = batch_graph.size(1)
        #
        # self.encoder.city_embed = self.encoder.city_embed[None, :].expand(expand_step, -1, -1, -1).reshape(B, N, -1)
        # self.encoder.nodes_embed = self.encoder.nodes_embed[None, :].expand(expand_step, -1, -1, -1).reshape(B, N, -1)
        # self.encoder.city_embed_mean = self.encoder.city_embed_mean[None, :].expand(expand_step, -1, -1, -1).reshape(B, 1,
        #                                                                                                           -1)

        agent_embed, V = self.encoder(agent_states, agents_city_mask=salesmen_mask)

        batch_indice = torch.arange(B, device=agent_states.device)[:, None]
        cur_pos = self.encoder.city_embed[torch.arange(agent_states.size(0))[:, None], agent_states[:, :, 1].long(), :]
        cur_pos_mean = cur_pos.mean(dim=1)
        actions_embed = torch.zeros((B, A, self.embed_dim), dtype=torch.float32, device=agent_states.device)
        actions_embed[:, 0, :] = cur_pos_mean
        actions_embed[:, 1:, :] = self.encoder.city_embed[batch_indice, act[..., :-1]]

        agents_mask = torch.triu(torch.ones(A, A), diagonal=1).to(agent_embed.device).bool()[None,].expand(B, A, A)
        act_logits = self.decoder(actions_embed, agent_embed,
                                  agents_mask, self.encoder.city_embed, salesmen_mask, n_agent = A)

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
