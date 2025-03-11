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

        self.lr = args.lr
        self.grad_max_norm = args.grad_max_norm
        self.act_optim = optim.AdamW(self.model.actions_model.parameters(), lr=self.lr)
        self.conf_optim = optim.AdamW(self.model.conflict_model.parameters(), lr=self.lr)
        self.optim = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() and self.args.use_gpu else "cpu")
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, "min", patience=10000 / args.eval_interval, factor= 0.99,min_lr=2e-5)
        self.model.to(self.device)

    def reset_graph(self, graph):
        """
        :param graph: [B,N,2]
        :return:
        """
        graph_t = _convert_tensor(graph, device=self.device, target_shape_dim=3)
        self.model.init_city(graph_t)

    def __get_action_logprob(self, states, masks, mode="greedy"):
        actions_logits, agents_logits, acts, acts_no_conflict, agents_mask = self.model(states, masks, {"mode": mode})
        actions_dist = torch.distributions.Categorical(logits=actions_logits)
        agents_dist = torch.distributions.Categorical(logits=agents_logits)
        act_logp = actions_dist.log_prob(acts)
        agents_logp = agents_dist.log_prob(agents_logits.argmax(dim=-1))
        agents_logp = torch.where(agents_mask, agents_logp, 0)
        agt_entropy = torch.where(agents_mask, agents_dist.entropy(), 0)
        return acts, acts_no_conflict, act_logp, agents_logp, actions_dist.entropy(), agt_entropy

    def predict(self, states_t, masks_t):
        self.model.train()
        actions, actions_no_conflict, act_logp, agents_logp, act_entropy, agt_entropy = self.__get_action_logprob(states_t, masks_t,
                                                                                        mode="sample")
        return actions.cpu().numpy(), actions_no_conflict.cpu().numpy(), act_logp, agents_logp, act_entropy, agt_entropy

    def exploit(self, states_t, masks_t, mode="greedy"):
        self.model.eval()
        actions, actions_no_conflict, _, _, _, _ = self.__get_action_logprob(states_t, masks_t, mode=mode)
        return actions.cpu().numpy(), actions_no_conflict.cpu().numpy()

    def __update_net(self, optim, params, loss):
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, self.grad_max_norm)
        optim.step()

    def __get_logprob(self, states, masks, actions):
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

    def __get_loss(self, act_logp, agents_logp, costs):

        # 智能体间平均， 组间最小化最大
        costs_8 = costs.reshape(costs.shape[0] // 8, 8, -1)  # 将成本按实例组进行分组
        act_logp_8 = act_logp.reshape(act_logp.shape[0] // 8, 8, -1)  # 将动作概率按实例组进行分组
        agents_logp_8 = agents_logp.reshape(agents_logp.shape[0] // 8, 8, -1)  # 将智能体动作概率按实例组进行分组

        agents_avg_cost = np.mean(costs_8, keepdims=True, axis=-1)
        agents_max_cost = np.max(costs_8, keepdims=True, axis=-1)
        # 智能体间优势
        agents_adv = np.abs(costs_8 - agents_avg_cost)
        agents_adv = (agents_adv - agents_adv.mean( keepdims=True,axis = -1))/(agents_adv.std(axis=-1, keepdims=True) + 1e-8)
        # agents_adv = (agents_adv - agents_adv.mean( keepdims=True,axis = -1))/(agents_adv.std(axis=-1, keepdims=True) + 1e-8)
        # 实例间优势
        # group_adv = agents_max_cost - np.min(agents_max_cost, keepdims=True, axis=1)
        group_adv = (agents_max_cost - np.mean(agents_max_cost, keepdims=True, axis=1)) / (agents_max_cost.std( keepdims=True, axis=1) + 1e-8)
        # 组合优势
        adv_actions = 0.5*agents_adv + group_adv
        adv_agents = 0.5*agents_adv  + group_adv

        # 转换为tensor并放到指定的device上
        adv_actions = _convert_tensor(adv_actions, device=self.device)
        adv_agents = _convert_tensor(adv_agents, device=self.device)

        # 对动作概率为零的样本进行掩码
        mask_ = ((act_logp_8 != 0) & (~torch.isnan(act_logp_8)))

        # 计算动作网络的损失，mask之后加权平均
        act_loss = (act_logp_8[mask_] * adv_actions[mask_]).mean()

        # 对智能体的动作概率进行掩码
        mask_ = ((agents_logp_8 != 0) & (~torch.isnan(agents_logp_8)))

        # 计算智能体的损失，mask之后加权平均
        agents_loss = (agents_logp_8[mask_] * adv_agents[mask_]).mean()

        return act_loss, agents_loss


    def learn(self, act_logp, agents_logp,act_ent, agt_ent, costs):
        self.model.train()
        act_loss, conflict_loss = self.__get_loss(act_logp, agents_logp, costs)
        act_ent_loss = act_ent
        if not torch.any(torch.isnan(conflict_loss)):
            agt_ent_loss = agt_ent
        else:
            conflict_loss = torch.tensor([0], device=self.device)
            agt_ent_loss = torch.tensor([0], device=self.device)
        self.__update_net(self.optim, self.model.parameters(), act_loss + conflict_loss + self.args.entropy_coef * (-act_ent_loss - agt_ent_loss))
        return act_loss.item(), conflict_loss.item(), act_ent_loss.item(), agt_ent_loss.item()

    def run_batch_episode(self, env, batch_graph, agent_num, eval_mode=False, exploit_mode="sample"):
        states, info = env.reset(
            config={
                "cities": batch_graph.shape[1],
                "salesmen": agent_num,
                "mode": "fixed",
                "N_aug": batch_graph.shape[0],
            },
            graph=batch_graph
        )
        salesmen_masks = info["salesmen_masks"]
        self.reset_graph(batch_graph)
        act_logp_list = []
        agents_logp_list = []
        act_ent_list = []
        agt_ent_list = []

        done = False
        info = None
        while not done:
            states_t = _convert_tensor(states, device=self.device)
            salesmen_masks_t = _convert_tensor(salesmen_masks, device=self.device)
            if eval_mode:
                acts, acts_no_conflict = self.exploit(states_t, salesmen_masks_t, exploit_mode)
            else:
                acts, acts_no_conflict, act_logp, agents_logp, act_entropy, agt_entropy = self.predict(states_t, salesmen_masks_t)
                act_logp_list.append(act_logp.unsqueeze(-1))
                agents_logp_list.append(agents_logp.unsqueeze(-1))
                act_ent_list.append(act_entropy.unsqueeze(-1))
                agt_ent_list.append(agt_entropy.unsqueeze(-1))
            states, r, done, info = env.step(acts_no_conflict + 1)
            salesmen_masks = info["salesmen_masks"]

        if eval_mode:
            return info
        else:
            act_logp = torch.cat(act_logp_list, dim=-1)
            agents_logp = torch.cat(agents_logp_list, dim=-1)
            act_ent = torch.cat(act_ent_list, dim=-1).mean()
            agt_ent = torch.cat(agt_ent_list, dim=-1)
            agt_ent = agt_ent.sum() / agt_ent.count_nonzero()

            # 将logp为0的部分权重为0
            act_logp = torch.where(act_logp == 0, 0.0, act_logp)  # logp为0时置为0
            agents_logp = torch.where(agents_logp == 0, 0.0, agents_logp)

            act_likelihood = torch.sum(act_logp, dim=-1)
            agents_likelihood = torch.sum(agents_logp, dim=-1)
            return (
                act_likelihood,
                agents_likelihood,
                act_ent,
                agt_ent,
                info["costs"]
            )

    def eval_episode(self, env, batch_graph, agent_num, exploit_mode="sample"):
        with torch.no_grad():
            eval_info = self.run_batch_episode(env, batch_graph, agent_num, eval_mode=True, exploit_mode=exploit_mode)
            cost = np.max(eval_info["costs"], axis=1)
            trajectory = eval_info["trajectories"]
            return cost, trajectory

    def state_dict(self):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_act_optim": self.act_optim.state_dict(),
            "model_conf_optim": self.conf_optim.state_dict(),
            "model_optim": self.optim.state_dict(),
        }
        return checkpoint

    def load_state_dict(self, checkpoint):
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.act_optim.load_state_dict(checkpoint["model_act_optim"])
        self.conf_optim.load_state_dict(checkpoint["model_conf_optim"])
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
