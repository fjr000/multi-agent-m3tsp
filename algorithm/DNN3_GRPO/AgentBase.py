import argparse

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from utils.TensorTools import _convert_tensor
import numpy as np


class AgentBase:
    def __init__(self, args, model_class):
        self.args = args
        self.model = model_class(agent_dim=args.agent_dim,
                           hidden_dim=args.hidden_dim,
                           embed_dim=args.embed_dim,
                           num_heads=args.num_heads,
                           num_layers=args.num_layers)
        self._gamma = args.gamma
        self.lr = args.lr
        self.grad_max_norm = args.grad_max_norm
        self.optim = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() and self.args.use_gpu else "cpu")
        # self.device = torch.device("cpu")
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
        actions, logp = self.__get_action_logprob(states_t, masks_t, mode="sample")
        return actions.cpu().numpy().squeeze(0), logp.cpu().numpy().squeeze(0)

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

    def get_cumulative_returns_batch(self, reward_multi_list):
        returns_numpy = np.zeros((len(reward_multi_list),len(reward_multi_list[0])))
        returns_numpy[-1] = reward_multi_list[-1]
        for idx in range(-2, -len(reward_multi_list) - 1, -1):
            returns_numpy[idx] = reward_multi_list[idx] + self._gamma * returns_numpy[idx + 1]
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

    def __get_likelihood(self, actions_logprob, dones):
        indice = dones.nonzero()[0]
        likeilihood = torch.empty((indice.shape[0],actions_logprob.size(1)), device=self.device)
        pre = 0
        for i in range(indice.shape[0]):
            clipSum = torch.sum(actions_logprob[pre:indice[i]], dim = 0)
            likeilihood[i] = clipSum
            pre = indice[i]+1
        return likeilihood

    def __get_loss(self, states, masks, actions, returns, reward, dones, act_logp):
        group_avg_reward = reward.mean()
        group_std_reward = reward.std() + 1e-8

        actions_logprob, entropy = self.__get_logprob(states, masks, actions)
        # likelihood = self.__get_likelihood(actions_logprob, dones);
        # Rs = returns[dones.nonzero()[0]]

        ratio = (actions_logprob - act_logp).exp()
        adv = (returns - group_avg_reward) / group_std_reward
        loss = - (ratio * adv).mean()

        if self.args.max_ent:
            loss -= self.args.entropy_coef * entropy.mean()

        return loss

    def learn(self, graph_t, states_tb, actions_tb, returns_tb, reward_nb, masks_tb, done_nb, logp_nb):
        self.model.train()
        self.model.init_city(graph_t)
        loss = self.__get_loss(states_tb, masks_tb, actions_tb, returns_tb, reward_nb, done_nb, logp_nb)
        self.__update_net(self.optim, self.model.parameters(), loss)
        self.model.init_city(graph_t)
        # del states_tb, actions_tb, returns_tb, masks_tb
        return loss.item()

    def _run_episode(self, env, graph, agent_num, eval_mode=False, exploit_mode = "sample"):
        agents_states, info = env.reset({"mode": "fixed"}, graph = graph[0])
        agents_mask = info["salesmen_masks"]
        done = False
        self.reset_graph(graph)
        states_list = []
        actions_list = []
        reward_list = []
        masks_list = []
        done_list = []
        logp_list = []
        with (torch.no_grad()):
            while not done:
                agents_states_t = _convert_tensor(agents_states, device=self.device, target_shape_dim=3)
                agents_mask_t = _convert_tensor(agents_mask, device=self.device, target_shape_dim=3)
                if eval_mode:
                    actions = self.exploit(agents_states_t, agents_mask_t, mode=exploit_mode)
                else:
                    actions, logp = self.predict(agents_states_t, agents_mask_t)

                states, rewards, done, info = env.step(actions + 1)

                if not eval_mode:
                    states_list.append(agents_states_t.squeeze(0))
                    masks_list.append(agents_mask_t.squeeze(0))
                    actions_list.append(actions)
                    reward_list.append(rewards)
                    done_list.append(done)
                    logp_list.append(logp)

                agents_mask = info["salesmen_masks"]
                agents_states = states

            if not eval_mode:
                returns_nb = self.get_cumulative_returns_batch(np.array(reward_list)[:,np.newaxis].repeat(agent_num, axis=1))
                reward_nb = np.array([reward_list[-1]])
                states_nb = torch.stack(states_list, dim=0).cpu().numpy()
                actions_nb = np.stack(actions_list, axis=0)
                logp_nb = np.stack(logp_list, axis=0)
                masks_nb = torch.stack(masks_list, dim=0).cpu().numpy()
                done_nb = np.stack(done_list, axis=0)
                return states_nb, actions_nb, returns_nb, reward_nb, masks_nb, done_nb, logp_nb
            else:
                return info

    def eval_episode(self, env, graph, agent_num, exploit_mode = "sample"):
        with torch.no_grad():
            eval_info = self._run_episode(env, graph, agent_num, eval_mode=True, exploit_mode=exploit_mode)
            cost = np.max(eval_info["costs"])
            trajectory = eval_info["trajectories"]
            return cost, trajectory

    def run_batch(self, env, graph, agent_num, batch_size):
        cur_size = 0
        states_nb_list = []
        actions_nb_list = []
        returns_nb_list = []
        reward_nb_list = []
        masks_nb_list = []
        done_nb_list = []
        logp_nb_list = []
        with torch.no_grad():
            while cur_size < batch_size:
                features_nb, actions_nb, returns_nb, reward_nb, masks_nb, done_nb, logp_nb = self._run_episode(env, graph, agent_num)
                states_nb_list.append(features_nb)
                actions_nb_list.append(actions_nb)
                returns_nb_list.append(returns_nb)
                reward_nb_list.append(reward_nb)
                masks_nb_list.append(masks_nb)
                done_nb_list.append(done_nb)
                logp_nb_list.append(logp_nb)
                cur_size += len(masks_nb)
        return (
            np.concatenate(states_nb_list, axis=0),
            np.concatenate(actions_nb_list, axis=0),
            np.concatenate(returns_nb_list, axis=0),
            np.concatenate(reward_nb_list, axis=0),
            np.concatenate(masks_nb_list, axis=0),
            np.concatenate(done_nb_list, axis=0),
            np.concatenate(logp_nb_list, axis=0),
        )

    def state_dict(self):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_optim": self.optim.state_dict(),
        }
        return checkpoint

    def load_state_dict(self, checkpoint):
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optim.load_state_dict(checkpoint["model_optim"])

    def _save_model(self, model_dir, filename):
        save_path = f"{model_dir}{filename}.pth"
        checkpoint = self.state_dict()
        torch.save(checkpoint, save_path)

    def _load_model(self, model_dir, filename):
        load_path = f"{model_dir}{filename}.pth"
        import os
        if os.path.isfile(load_path):
            checkpoint = torch.load(load_path, weights_only=False, map_location=self.device)
            self.load_state_dict(checkpoint)
            print(f"load {load_path} successfully")
        else:
            print("model file doesn't exist")
