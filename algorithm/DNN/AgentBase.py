import argparse

from model.model_v1 import Model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from utils.TensorTools import _convert_tensor
import numpy as np


class AgentBase:
    def __init__(self, args):
        self.args = args
        self.model = Model(agent_dim=args.agent_dim,
                           hidden_dim=args.hidden_dim,
                           embed_dim=args.embed_dim,
                           num_heads=args.num_heads,
                           num_layers=args.num_layers)
        self._gamma = args.gamma
        self.lr = args.lr
        self.grad_max_norm = args.grad_max_norm
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        self.device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def reset_graph(self, graph):
        """
        :param graph: [B,N,2]
        :return:
        """
        graph_t = _convert_tensor(graph, device=self.device, target_shape_dim=3)
        self.model.init_city(graph_t)
        # print(f"city_embed:{self.model.city_embed[0,0,:2]}")

    def __get_action_logprob(self, states, masks, mode="greedy"):
        actions_logits = self.model(states, masks)
        dist = torch.distributions.Categorical(logits=actions_logits)
        if mode == "greedy":
            action = torch.argmax(actions_logits, dim=-1)
            logprob = dist.log_prob(action)
        elif mode == "sample":
            action = dist.sample()
            logprob = dist.log_prob(action)
        else:
            raise NotImplementedError
        return action, logprob

    def predict(self, states_t, masks_t):
        self.model.train()
        actions, _ = self.__get_action_logprob(states_t, masks_t, mode="sample")
        return actions.cpu().numpy().squeeze(0)

    def exploit(self, states_t, masks_t, mode="greedy"):
        self.model.eval()
        actions, _ = self.__get_action_logprob(states_t, masks_t, mode=mode)
        return actions.cpu().numpy().squeeze(0)

    def get_cumulative_returns(self, reward_list):
        returns_numpy = np.zeros(len(reward_list))
        returns_numpy[-1] = reward_list[-1]
        for idx in range(-2, -len(reward_list) - 1, -1):
            returns_numpy[idx] = reward_list[idx] + self._gamma * returns_numpy[idx + 1]
        return returns_numpy

    def __update_net(self, optim, params, loss):
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, self.grad_max_norm)
        optim.step()

    def __get_logprob(self, states, masks, actions):
        actions_logits = self.model(states, masks)
        dist = torch.distributions.Categorical(logits=actions_logits)
        entropy = dist.entropy()
        return dist.log_prob(actions), entropy

    def __get_loss(self, states, masks, actions, returns):
        actions_logprob, entropy = self.__get_logprob(states, masks, actions)

        if self.args.returns_norm:
            loss = - (actions_logprob * (returns - returns.mean())).mean()
        else:
            loss = - (actions_logprob * returns).mean()

        if self.args.max_ent:
            loss -= self.args.entropy_coef * entropy.mean()

        return loss

    def learn(self, graph_t, states_tb, actions_tb, returns_tb, masks_tb):
        self.model.train()
        self.model.init_city(graph_t)
        loss = self.__get_loss(states_tb, masks_tb, actions_tb, returns_tb)
        self.__update_net(self.optim, self.model.parameters(), loss)
        self.model.init_city(graph_t)
        return loss.item()

    def _run_episode(self, env, graph, agent_num, eval_mode = False):
        env.init_fixed_graph(graph, agent_num)
        agents_states, info = env.reset({"fixed_graph": True})
        agents_mask = info["agents_mask"]
        agents_last_states = info["actors_last_states"]

        done = False
        self.reset_graph(graph)
        last_states_list = []
        states_list = []
        actions_list = []
        reward_list = []
        masks_list = []
        while not done:
            agents_states_t = _convert_tensor(agents_states, device=self.device, target_shape_dim=3)
            last_states_t = _convert_tensor(agents_last_states, device=self.device, target_shape_dim=3)
            agents_mask_t = _convert_tensor(agents_mask, device=self.device, target_shape_dim=3)
            if eval_mode:
                actions = self.exploit([last_states_t, agents_states_t], agents_mask_t)
            else:
                actions = self.predict([last_states_t, agents_states_t], agents_mask_t)

            states, reward, done, info = env.step(actions + 1)

            if not eval_mode:
                last_states_list.append(last_states_t)
                states_list.append(agents_states_t)
                masks_list.append(agents_mask_t)
                actions_list.append(torch.tensor(actions, dtype=torch.float32, device=self.device).unsqueeze(0))
                reward_list.append(reward)

            agents_mask = info["agents_mask"]
            agents_last_states = info["actors_last_states"]
            agents_states = states

        if not eval_mode:
            returns_numpy = self.get_cumulative_returns(reward_list)
            states_tb = torch.cat(states_list, dim=0)
            last_states_tb = torch.cat(last_states_list, dim=0)
            actions_tb = torch.cat(actions_list, dim=0)
            returns_tb = torch.tensor(np.repeat(returns_numpy[:, np.newaxis], agent_num, axis=1), dtype=torch.float32,
                                      device=self.device)
            masks_tb = torch.cat(masks_list, dim=0)
            return [last_states_tb, states_tb], actions_tb, returns_tb, masks_tb
        else:
            return info

    def eval_episode(self, env, graph, agent_num):
        eval_info = self._run_episode(env, graph, agent_num, eval_mode = True)
        return eval_info

    def run_batch(self, env, graph, agent_num, batch_size):
        cur_size = 0
        last_states_tb_list = []
        states_tb_list = []
        actions_tb_list = []
        returns_tb_list = []
        masks_tb_list = []
        while cur_size < batch_size:
            features_tb, actions_tb, returns_tb, masks_tb = self._run_episode(env, graph, agent_num)
            last_states_tb_list.append(features_tb[0])
            states_tb_list.append(features_tb[1])
            actions_tb_list.append(actions_tb)
            returns_tb_list.append(returns_tb)
            masks_tb_list.append(masks_tb)
            cur_size += masks_tb.size(0)

        return (
                [torch.cat(last_states_tb_list, dim=0), torch.cat(states_tb_list, dim=0)],
                torch.cat(actions_tb_list, dim=0),
                torch.cat(returns_tb_list, dim=0),
                torch.cat(masks_tb_list, dim=0),
                )
