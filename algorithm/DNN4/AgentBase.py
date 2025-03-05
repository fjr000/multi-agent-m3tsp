import argparse

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from utils.TensorTools import _convert_tensor
import numpy as np


class AgentBase:
    def __init__(self, args, config, model_class):
        self.args = args
        self.model = model_class(config)
        self.conflict_model = None

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

    def __get_action_logprob(self, states, masks, mode="greedy"):
        actions_logits, agents_logits, acts, acts_no_conflict = self.model(states, masks, {"mode": mode})
        actions_dist = torch.distributions.Categorical(logits=actions_logits)
        agents_dist = torch.distributions.Categorical(logits=agents_logits)
        act_logp = actions_dist.log_prob(acts)
        agents_logp = agents_dist.log_prob(agents_logits.argmax(dim=-1))
        return acts, acts_no_conflict, act_logp, agents_logp

    def predict(self, states_t, masks_t):
        self.model.train()
        actions, actions_no_conflict, _, _ = self.__get_action_logprob(states_t, masks_t, mode="sample")
        return actions.cpu().numpy().squeeze(0), actions_no_conflict.cpu().numpy().squeeze(0)

    def exploit(self, states_t, masks_t, mode="greedy"):
        self.model.eval()
        actions, actions_no_conflict, _, _ = self.__get_action_logprob(states_t, masks_t, mode=mode)
        return actions.cpu().numpy().squeeze(0), actions_no_conflict.cpu().numpy().squeeze(0)

    def get_cumulative_returns(self, reward_list):
        returns_numpy = np.zeros(len(reward_list))
        returns_numpy[-1] = reward_list[-1]
        for idx in range(-2, -len(reward_list) - 1, -1):
            returns_numpy[idx] = reward_list[idx] + self._gamma * returns_numpy[idx + 1]
        return returns_numpy

    def get_cumulative_returns_batch(self, reward_multi_list):
        returns_numpy = np.zeros((len(reward_multi_list), len(reward_multi_list[0])))
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
        # TODO: 是否需要actions传入model计算agents_logits
        actions_logits, agents_logits, acts, acts_no_conflict = self.model(states, masks)
        dist = torch.distributions.Categorical(logits=actions_logits)
        agents_dist = torch.distributions.Categorical(logits=agents_logits)
        agents_logp = agents_dist.log_prob(agents_logits.argmax(dim=-1))
        entropy = dist.entropy()
        return dist.log_prob(actions), entropy, agents_logp

    def __get_likelihood(self, actions_logprob, dones):
        indice = dones.nonzero()[0]
        likeilihood = torch.empty((indice.shape[0], actions_logprob.size(1)), device=self.device)
        pre = 0
        for i in range(indice.shape[0]):
            clipSum = torch.sum(actions_logprob[pre:indice[i]], dim=0)
            likeilihood[i] = clipSum
            pre = indice[i] + 1
        return likeilihood

    def __get_baseline(self, returns, dones):
        # 步骤1：根据dones拆分轨迹
        indices = dones.nonzero()[0]
        trajectories = []
        start = 0
        returns_nb = returns.cpu().numpy()
        for end in indices:
            trajectories.append(returns_nb[start:end + 1, :])  # 提取轨迹片段（包含结束时间步）
            start = end + 1

        # 步骤2：确定最大轨迹长度
        max_length = max([traj.shape[0] for traj in trajectories])

        # 步骤3：初始化基线数组（一维，按时间步）
        baseline_t = np.zeros(max_length)
        counts_t = np.zeros(max_length, dtype=np.int32)

        # 步骤4：向量化计算每个时间步的累积回报和计数
        for traj in trajectories:
            T = traj.shape[0]  # 当前轨迹的时间步数
            # 计算每个时间步所有智能体的回报总和
            sum_per_timestep = traj.sum(axis=1)  # 形状 [T]
            # 更新基线累积和计数
            baseline_t[:T] += sum_per_timestep
            counts_t[:T] += np.count_nonzero(traj, axis=1)

        # 步骤5：计算最终基线（处理除零）
        baseline_t = np.divide(
            baseline_t,
            counts_t,
            out=np.zeros_like(baseline_t),
            where=counts_t != 0
        )

        return torch.tensor(baseline_t, dtype=torch.float32, device=self.device)

    def __align_trajectories(self, x: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """
        将形状 [B, A] 的张量切割为 [max_T, traj_num*A]，保留梯度
        Args:
            x: 输入张量，形状 [B, A]
            dones: 轨迹终止标记，形状 [B]
        Returns:
            aligned_x: 对齐后的张量，形状 [max_T, traj_num*A]
        """
        # 步骤1：确定轨迹的结束位置
        end_indices = torch.nonzero(dones).squeeze(-1)

        # 计算轨迹分割点（每条轨迹的长度）
        split_indices = []
        start = 0
        for end in end_indices:
            split_indices.append(end - start + 1)  # 当前轨迹长度
            start = end + 1

        # 步骤2：按轨迹分割张量
        trajectories = torch.split(x, split_indices, dim=0)  # 列表，每个元素形状 [T_i, 5]

        # 步骤3：填充轨迹到统一长度 max_T
        import torch.nn.utils.rnn as rnn_utils
        padded_trajectories = rnn_utils.pad_sequence(
            trajectories,
            batch_first=False,  # 输出形状 [max_T, traj_num, 5]
            padding_value=0.0
        )  # 形状 [max_T, traj_num, 5]

        # 步骤4：合并轨迹和智能体维度 [max_T, traj_num*5]
        max_T, traj_num, A = padded_trajectories.shape
        aligned_x = padded_trajectories.reshape(max_T, traj_num * A)

        return aligned_x

    def __get_loss(self, states, masks, actions, returns, dones):
        actions_logprob, entropy, agents_logp = self.__get_logprob(states, masks, actions)
        # likelihood = self.__get_likelihood(actions_logprob, dones);
        # Rs = returns[dones.nonzero()[0]]
        bs = self.__get_baseline(returns, dones)
        logp = self.__align_trajectories(actions_logprob, torch.tensor(dones, dtype=torch.bool,device=self.device))
        align_returns = self.__align_trajectories(returns,torch.tensor(dones, dtype=torch.bool,device=self.device))
        as_logp = self.__align_trajectories(agents_logp, torch.tensor(dones, dtype=torch.bool,device=self.device))
        adv = (align_returns - bs.unsqueeze(1).expand(-1, align_returns.size(1)))
        # adv = returns - returns.mean()
        # if self.args.returns_norm:
        #     adv = adv / (returns.std() + 1e-8)
        loss = - (logp * adv).mean()

        if self.args.max_ent:
            loss -= self.args.entropy_coef * entropy.mean()

        loss += -(as_logp * adv).mean()

        return loss

    def learn(self, graph_t, states_tb, actions_tb, returns_tb, masks_tb, done_nb):
        self.model.train()
        self.model.init_city(graph_t)
        loss = self.__get_loss(states_tb, masks_tb, actions_tb, returns_tb, done_nb)
        self.__update_net(self.optim, self.model.parameters(), loss)
        self.model.init_city(graph_t)
        # del states_tb, actions_tb, returns_tb, masks_tb
        return loss.item()

    def _run_episode(self, env, graph, agent_num, eval_mode=False, exploit_mode="sample"):
        agents_states, info = env.reset({"mode": "fixed"}, graph=graph[0])
        agents_mask = info["salesmen_masks"]
        done = False
        self.reset_graph(graph)
        states_list = []
        actions_list = []
        actions_no_conflict_list = []
        reward_list = []
        individual_rewards_list = []
        masks_list = []
        done_list = []
        with torch.no_grad():
            while not done:
                agents_states_t = _convert_tensor(agents_states, device=self.device, target_shape_dim=3)
                agents_mask_t = _convert_tensor(agents_mask, device=self.device, target_shape_dim=3)
                if eval_mode:
                    actions, actions_no_conflict = self.exploit(agents_states_t, agents_mask_t, mode=exploit_mode)
                else:
                    actions, actions_no_conflict = self.predict(agents_states_t, agents_mask_t)

                states, rewards, done, info = env.step(actions_no_conflict + 1)

                if not eval_mode:
                    states_list.append(agents_states_t.squeeze(0))
                    masks_list.append(agents_mask_t.squeeze(0))
                    actions_list.append(actions)
                    actions_no_conflict_list.append(actions_no_conflict)
                    reward_list.append(rewards)
                    individual_rewards_list.append(info["individual_rewards"])
                    done_list.append(done)

                agents_mask = info["salesmen_masks"]
                agents_states = states

            if not eval_mode:
                returns_nb = self.get_cumulative_returns_batch(
                    np.array(reward_list)[:, np.newaxis].repeat(agent_num, axis=1))
                individual_returns_nb = returns_nb
                individual_returns_nb = np.array(individual_rewards_list)
                dones_step = info["dones_step"]
                costs = info["costs"]
                max_cost_id = np.argmax(info["costs"])
                max_cost = costs[max_cost_id]
                min_cost = np.min(costs)
                for i in range(agent_num):
                    individual_returns_nb[dones_step[i]-1,i] += (costs[i] - max_cost)

                individual_returns_nb[ dones_step[max_cost_id]-1, max_cost_id] -= (max_cost - min_cost)

                individual_returns_nb = self.get_cumulative_returns_batch(
                    individual_returns_nb
                )

                states_nb = torch.stack(states_list, dim=0).cpu().numpy()
                actions_nb = np.stack(actions_list, axis=0)
                actions_no_conflict_nb = np.stack(actions_no_conflict_list, axis=0)
                masks_nb = torch.stack(masks_list, dim=0).cpu().numpy()
                done_nb = np.stack(done_list, axis=0)
                return states_nb, actions_nb, actions_no_conflict_nb, returns_nb, individual_returns_nb, masks_nb, done_nb
            else:
                return info

    def eval_episode(self, env, graph, agent_num, exploit_mode="sample"):
        with torch.no_grad():
            eval_info = self._run_episode(env, graph, agent_num, eval_mode=True, exploit_mode=exploit_mode)
            cost = np.max(eval_info["costs"])
            trajectory = eval_info["trajectories"]
            return cost, trajectory

    def run_batch(self, env, graph, agent_num, batch_size):
        cur_size = 0
        states_nb_list = []
        actions_nb_list = []
        actions_no_conflict_nb_list = []
        returns_nb_list = []
        individual_returns_nb_list = []
        masks_nb_list = []
        done_nb_list = []
        with torch.no_grad():
            while cur_size < batch_size:
                features_nb, actions_nb, actions_no_conflict_nb, returns_nb, individual_returns_nb, masks_nb, done_nb = self._run_episode(env, graph, agent_num)
                states_nb_list.append(features_nb)
                actions_nb_list.append(actions_nb)
                actions_no_conflict_nb_list.append(actions_no_conflict_nb)
                returns_nb_list.append(returns_nb)
                individual_returns_nb_list.append(individual_returns_nb)
                masks_nb_list.append(masks_nb)
                done_nb_list.append(done_nb)
                cur_size += len(masks_nb)
        return (
            np.concatenate(states_nb_list, axis=0),
            np.concatenate(actions_nb_list, axis=0),
            np.concatenate(actions_no_conflict_nb_list, axis=0),
            np.concatenate(returns_nb_list, axis=0),
            np.concatenate(individual_returns_nb_list, axis=0),
            np.concatenate(masks_nb_list, axis=0),
            np.concatenate(done_nb_list, axis=0),
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
