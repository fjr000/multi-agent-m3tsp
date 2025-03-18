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
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, "min",
                                                                       patience=10000 / args.eval_interval, factor=0.99,
                                                                       min_lr=2e-5)
        self.model.to(self.device)
        self.train_count = 0

    def reset_graph(self, graph):
        """
        :param graph: [B,N,2]
        :return:
        """
        graph_t = _convert_tensor(graph, device=self.device, target_shape_dim=3)
        self.model.init_city(graph_t)

    def __get_action_logprob(self, states, masks, mode="greedy", info=None):
        info = {} if info is None else info
        info.update({
            "mode": mode,
        })
        actions_logits, agents_logits, acts, acts_no_conflict, agents_mask = self.model(states, masks, info)
        actions_dist = torch.distributions.Categorical(logits=actions_logits)
        act_logp = actions_dist.log_prob(acts)

        if agents_logits is not None:
            agents_dist = torch.distributions.Categorical(logits=agents_logits)
            agents_logp = agents_dist.log_prob(agents_logits.argmax(dim=-1))
            agents_logp = torch.where(agents_mask, agents_logp, 0)
            agt_entropy = torch.where(agents_mask, agents_dist.entropy(), 0)
        else:
            agents_logp = None
            agt_entropy = None

        return acts, acts_no_conflict, act_logp, agents_logp, actions_dist.entropy(), agt_entropy

    def predict(self, states_t, masks_t, info=None):
        self.model.train()
        actions, actions_no_conflict, act_logp, agents_logp, act_entropy, agt_entropy = self.__get_action_logprob(
            states_t, masks_t,
            mode="sample", info=info)
        return actions.cpu().numpy(), actions_no_conflict.cpu().numpy(), act_logp, agents_logp, act_entropy, agt_entropy

    def exploit(self, states_t, masks_t, mode="greedy", info=None):
        self.model.eval()
        actions, actions_no_conflict, _, _, _, _ = self.__get_action_logprob(states_t, masks_t, mode=mode, info=info)
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

        agents_avg_cost = np.mean(costs_8, keepdims=True, axis=-1)
        agents_max_cost = np.max(costs_8, keepdims=True, axis=-1)
        # 智能体间优势
        agents_adv = np.abs(costs_8 - agents_avg_cost)
        agents_adv = (agents_adv - agents_adv.mean(keepdims=True, axis=-1)) / (
                agents_adv.std(axis=-1, keepdims=True) + 1e-8)
        # agents_adv = (agents_adv - agents_adv.mean( keepdims=True,axis = -1))/(agents_adv.std(axis=-1, keepdims=True) + 1e-8)
        # 实例间优势
        # group_adv = agents_max_cost - np.min(agents_max_cost, keepdims=True, axis=1)
        group_adv = (agents_max_cost - np.mean(agents_max_cost, keepdims=True, axis=1)) / (
                agents_max_cost.std(keepdims=True, axis=1) + 1e-8)
        # 组合优势
        adv = self.args.agents_adv_rate * agents_adv + group_adv

        # 转换为tensor并放到指定的device上
        adv_t = _convert_tensor(adv, device=self.device)

        # 对动作概率为零的样本进行掩码
        mask_ = ((act_logp_8 != 0) & (~torch.isnan(act_logp_8)))

        # 计算动作网络的损失，mask之后加权平均
        act_loss = (act_logp_8[mask_] * adv_t[mask_]).mean()
        if agents_logp is not None:
            agents_logp_8 = agents_logp.reshape(agents_logp.shape[0] // 8, 8, -1)  # 将智能体动作概率按实例组进行分组
            # 对智能体的动作概率进行掩码
            mask_ = ((agents_logp_8 != 0) & (~torch.isnan(agents_logp_8)))

            # 计算智能体的损失，mask之后加权平均
            agents_loss = (agents_logp_8[mask_] * adv_t[mask_]).mean()
        else:
            agents_loss = None

        return act_loss, agents_loss

    def __get_loss_only_instance(self, act_logp, agents_logp, costs):
        # 智能体间平均， 组间最小化最大

        agents_avg_cost = np.mean(costs, keepdims=True, axis=-1)
        agents_max_cost = np.max(costs, keepdims=True, axis=-1)
        # 智能体间优势
        agents_adv = np.abs(costs - agents_avg_cost)
        # agents_adv = agents_adv - agents_adv.mean(keepdims=True, axis=-1)
        agents_adv = (agents_adv - agents_adv.mean(keepdims=True, axis=-1)) / (
                    agents_adv.std(axis=-1, keepdims=True) + 1e-8)
        # 实例间优势
        # group_adv = agents_max_cost - np.min(agents_max_cost, keepdims=True, axis=1)
        group_adv = (agents_max_cost - np.mean(agents_max_cost, keepdims=True, axis=0)) / (
                    agents_max_cost.std(keepdims=True, axis=0) + 1e-8)
        # 组合优势
        act_adv = self.args.agents_adv_rate * agents_adv + group_adv
        agt_adv = self.args.agents_adv_rate * agents_adv + group_adv

        # 转换为tensor并放到指定的device上
        act_adv_t = _convert_tensor(act_adv, device=self.device)
        agt_adv_t = _convert_tensor(agt_adv, device=self.device)

        # 对动作概率为零的样本进行掩码
        mask_ = ((act_logp != 0) & (~torch.isnan(act_logp)))

        # 计算动作网络的损失，mask之后加权平均
        act_loss = (act_logp[mask_] * act_adv_t[mask_]).mean()
        if agents_logp is not None:
            # 对智能体的动作概率进行掩码
            mask_ = ((agents_logp != 0) & (~torch.isnan(agents_logp)))

            # 计算智能体的损失，mask之后加权平均
            agents_loss = (agents_logp[mask_] * agt_adv_t[mask_]).mean()
        else:
            agents_loss = None

        return act_loss, agents_loss

    def learn(self, act_logp, agents_logp, act_ent, agt_ent, costs):
        self.model.train()
        self.train_count += 1

        loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        agt_ent_loss = torch.tensor([0], device=self.device)
        agents_loss = torch.tensor([0], device=self.device)

        if self.args.only_one_instance:
            act_loss, agents_loss = self.__get_loss_only_instance(act_logp, agents_logp, costs)
        else:
            act_loss, agents_loss = self.__get_loss(act_logp, agents_logp, costs)
        act_ent_loss = act_ent

        if agents_logp is not None and self.args.train_conflict_model:
            # 修改为检查 agents_loss 是否包含 NaN
            if not torch.any(torch.isnan(agents_loss)):
                agt_ent_loss = agt_ent
            else:
                agents_loss = torch.tensor([0], device=self.device)
                agt_ent_loss = torch.tensor([0], device=self.device)

            # if not torch.isnan(agt_ent_loss):
                # 更新损失计算，确保使用正确的变量名称
            loss += self.args.conflict_loss_rate * agents_loss + self.args.entropy_coef * (- agt_ent_loss)

        if self.args.train_actions_model:
            loss += act_loss + self.args.entropy_coef * (- act_ent_loss)
        else:
            act_ent_loss = torch.tensor([0], device=self.device)
            act_ent_loss = torch.tensor([0], device=self.device)

        if not torch.isnan(agt_ent_loss):
            loss /= self.args.accumulation_steps
            loss.backward()

        if self.train_count % self.args.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_max_norm)
            self.optim.step()
            self.optim.zero_grad()
        return act_loss.item(), agents_loss.item(), act_ent_loss.item(), agt_ent_loss.item()



    def run_batch_episode(self, env, batch_graph, agent_num, eval_mode=False, exploit_mode="sample", info=None):
        states, env_info = env.reset(
            config={
                "cities": batch_graph.shape[1],
                "salesmen": agent_num,
                "mode": "fixed",
                "N_aug": batch_graph.shape[0],
                "use_conflict_model":info.get("use_conflict_model", False) if info is not None else False,
            },
            graph=batch_graph
        )
        salesmen_masks = env_info["salesmen_masks"]
        masks_in_salesmen = env_info["masks_in_salesmen"]
        city_mask = env_info["mask"]

        self.reset_graph(batch_graph)
        act_logp_list = []
        agents_logp_list = []
        act_ent_list = []
        agt_ent_list = []

        done = False
        use_conflict_model = False
        while not done:
            states_t = _convert_tensor(states, device=self.device)
            # mask: true :not allow  false:allow

            salesmen_masks_t = _convert_tensor(~salesmen_masks, dtype= torch.bool, device=self.device)
            masks_in_salesmen_t = _convert_tensor(~masks_in_salesmen, dtype= torch.bool, device=self.device)
            city_mask_t = _convert_tensor(~city_mask, dtype= torch.bool, device=self.device)
            info = {} if info is None else info
            info.update({
                "masks_in_salesmen":masks_in_salesmen_t,
                "mask":city_mask_t
            })
            if eval_mode:
                acts, acts_no_conflict = self.exploit(states_t, salesmen_masks_t, exploit_mode, info)
            else:
                acts, acts_no_conflict, act_logp, agents_logp, act_entropy, agt_entropy = self.predict(states_t,
                                                                                                       salesmen_masks_t,
                                                                                                       info)
                act_logp_list.append(act_logp.unsqueeze(-1))
                act_ent_list.append(act_entropy.unsqueeze(-1))
                if agents_logp is not None:
                    use_conflict_model = True
                    agents_logp_list.append(agents_logp.unsqueeze(-1))
                    agt_ent_list.append(agt_entropy.unsqueeze(-1))
            states, r, done, env_info = env.step(acts_no_conflict + 1)
            salesmen_masks = env_info["salesmen_masks"]
            masks_in_salesmen = env_info["masks_in_salesmen"]
            city_mask = env_info["mask"]

        if eval_mode:
            return env_info
        else:
            act_logp = torch.cat(act_logp_list, dim=-1)
            act_ent = torch.cat(act_ent_list, dim=-1).mean()
            act_ent = act_ent.sum() / act_ent.count_nonzero()
            # act_logp = torch.where(act_logp == 0, 0.0, act_logp)  # logp为0时置为0
            act_likelihood = torch.sum(act_logp, dim=-1)

            if use_conflict_model:
                agents_logp = torch.cat(agents_logp_list, dim=-1)
                agt_ent = torch.cat(agt_ent_list, dim=-1)
                agt_ent = agt_ent.sum() / agt_ent.count_nonzero()
                # agents_logp = torch.where(agents_logp == 0, 0.0, agents_logp)
                agents_likelihood = torch.sum(agents_logp, dim=-1)
            else:
                agents_likelihood = None
                agt_ent = None

            # act_likelihood = torch.sum(act_logp, dim=-1) / act_logp.count_nonzero(dim = -1)
            # agents_likelihood = torch.sum(agents_logp, dim=-1) / agents_logp.count_nonzero(dim=-1)
            return (
                act_likelihood,
                agents_likelihood,
                act_ent,
                agt_ent,
                env_info["costs"]
            )

    def eval_episode(self, env, batch_graph, agent_num, exploit_mode="sample", info=None):
        with torch.no_grad():
            eval_info = self.run_batch_episode(env, batch_graph, agent_num, eval_mode=True, exploit_mode=exploit_mode,
                                               info=info)
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
