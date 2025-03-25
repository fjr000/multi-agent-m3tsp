import argparse
from tabnanny import check

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from sympy import gamma

from utils.TensorTools import _convert_tensor
import numpy as np


class AgentBase:
    def __init__(self, args, config, model_class):
        self.args = args
        self.model = model_class(config)
        self.conflict_model = None

        self.lr = args.lr
        self.grad_max_norm = args.grad_max_norm
        # self.act_optim = optim.AdamW(self.model.actions_model.parameters(), lr=self.lr)
        # self.conf_optim = optim.AdamW(self.model.conflict_model.parameters(), lr=self.lr)
        self.optim = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() and self.args.use_gpu else "cpu")
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, "min",
                                                                       patience=10000 / args.eval_interval, factor=0.99,
                                                                       min_lr=2e-5)
        self.model.to(self.device)
        self.train_count = 0
        self._gamma = 1
        self._lambda = 0.95
        self.clip = 0.2
        self.value_loss_coef = 0.5
        self.entropy_loss_coef = 0.005

    def reset_graph(self, graph):
        """
        :param graph: [B,N,2]
        :return:
        """
        graph_t = _convert_tensor(graph, device=self.device, target_shape_dim=3)
        self.model.init_city(graph_t)

    def __get_action_logprob(self, states, salesmen_mask, mode="greedy"):
        act_logits, act, act_mask, V = self.model(states, salesmen_mask=salesmen_mask, mode=mode)

        dist = torch.distributions.Categorical(logits=act_logits)
        act_logp = dist.log_prob(act)

        return act_logp, act, act_logp, act_mask, dist.entropy(), V

    def predict(self, states_t, masks_t):
        self.model.eval()
        act_logp, act, act_logp, act_mask, entropy, V = self.__get_action_logprob(
            states_t, masks_t,
            mode="sample")
        return act.cpu().numpy(), act_logp.cpu().numpy(), entropy.cpu().numpy(), act_mask.cpu().numpy(), V.cpu().numpy()

    def exploit(self, states_t, masks_t, mode="greedy"):
        self.model.eval()
        act_logp, act, act_logp, act_mask, entropy, V = self.__get_action_logprob(
            states_t, masks_t,
            mode=mode)
        return act.cpu().numpy()

    def get_value(self, states_t, masks_t):
        return self.model.get_value(states_t, masks_t)
    def __update_net(self, optim, params, loss):
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, self.grad_max_norm)
        optim.step()

    def __eval_logprob_val(self, states, salesmen_mask, actions, batch_graph, expand_step):
        act_logits, V = self.model(states, salesmen_mask=salesmen_mask, act = actions,
                                                  batch_graph = batch_graph, expand_step = expand_step)

        dist = torch.distributions.Categorical(logits=act_logits)
        act_logp = dist.log_prob(actions)

        return act_logp, dist.entropy(), V

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
        adv = self.args.agents_adv_rate*agents_adv + group_adv

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
        agents_adv = (agents_adv - agents_adv.mean( keepdims=True,axis = -1))/(agents_adv.std(axis=-1, keepdims=True) + 1e-8)
        # 实例间优势
        # group_adv = agents_max_cost - np.min(agents_max_cost, keepdims=True, axis=1)
        group_adv = (agents_max_cost - np.mean(agents_max_cost, keepdims=True, axis=0)) / (agents_max_cost.std(keepdims=True, axis=0) + 1e-8)
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

    def _get_policy_loss(self, act_logp_new, act_logp, adv, mask):
        ratio = torch.exp(act_logp_new - act_logp)[mask]

        adv = adv[mask]

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 -self.clip, 1.0 + self.clip) * adv
        loss = -  torch.min(surr1, surr2).mean()

        return loss

    def _get_value_loss(self, returns, v_new, v, mask):
        v_clip = v + (v_new - v).clamp(-self.clip, self.clip)
        value_loss_clipped = F.mse_loss(returns[mask], v_clip[mask])
        value_loss_original = F.mse_loss(returns[mask], v_new[mask])

        value_loss = torch.max(value_loss_clipped, value_loss_original)

        return value_loss

    def check(self, input, dtype):
        return _convert_tensor(input, dtype = dtype, device=self.device)

    def learn(self,batch_graph, states, act_logp, act, act_ent, act_mask, gae, returns, V, expand_step, cost):
        self.model.train()
        self.train_count += 1

        states_t = self.check(states, dtype = torch.float32)
        batch_graph_t = self.check(batch_graph, dtype = torch.float32)
        act_logp_t = self.check(act_logp, dtype = torch.float32)
        act_t = self.check(act, dtype = torch.int)
        act_mask_t = self.check(act_mask, dtype = torch.bool)
        gae_t = self.check(gae, dtype = torch.float32)
        returns_t = self.check(returns, dtype = torch.float32)
        V_t = self.check(V, dtype = torch.float32)

        mask = ~torch.isclose(act_logp_t, torch.zeros_like(act_logp_t), rtol=1e-8, atol= 1e-10)

        act_logp_new, entropy_new, V_new = self.__eval_logprob_val(states_t, act_mask_t, act_t, batch_graph_t, expand_step)

        policy_loss = self._get_policy_loss(act_logp_new, act_logp_t, gae_t, mask)

        value_loss = self._get_value_loss(returns_t, V_new, V_t, mask)
        entropy_new_mask = entropy_new[mask]
        entropy_loss = entropy_new_mask.mean()

        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_loss_coef * entropy_loss

        loss /= self.args.accumulation_steps
        loss.backward()

        if self.train_count % self.args.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_max_norm)
            self.optim.step()
            self.optim.zero_grad()

        return policy_loss.item(), value_loss.item(), entropy_loss.item()

    def run_batch_episode(self, env, batch_graph, agent_num, eval_mode=False, exploit_mode="sample"):
        states, env_info = env.reset(
            config={
                "cities": batch_graph.shape[1],
                "salesmen": agent_num,
                "mode": "fixed",
                "N_aug": batch_graph.shape[0],
            },
            graph=batch_graph
        )
        salesmen_masks = env_info["salesmen_masks"]

        self.reset_graph(batch_graph)

        state_list = [states.copy(),]
        act_list = []
        act_logp_list = []
        act_mask_list = []
        act_ent_list = []
        V_list = []
        r_lsit = []
        act, act_logp, entropy, act_mask, V = None, None, None, None, None
        done = False
        expand_step = 0
        with torch.no_grad():
            while not done:
                states_t = _convert_tensor(states, device=self.device)
                # mask: true :not allow  false:allow
                salesmen_masks_t = _convert_tensor(~salesmen_masks, dtype= torch.bool, device=self.device)
                if eval_mode:
                    act = self.exploit(states_t, salesmen_masks_t, exploit_mode)
                else:
                    act, act_logp, entropy, act_mask, V = self.predict(states_t, salesmen_masks_t,)
                states, r, done, env_info = env.step(act + 1)

                if not eval_mode:
                    act_logp_list.append(act_logp)
                    act_ent_list.append(entropy)
                    V_list.append(V)
                    act_mask_list.append(act_mask)
                    act_list.append(act)
                    r_lsit.append(r)
                    if not done:
                        state_list.append(states.copy())

                salesmen_masks = env_info["salesmen_masks"]
                expand_step += 1

                if done:
                    states_t = _convert_tensor(states, device=self.device)
                    salesmen_masks_t = _convert_tensor(~salesmen_masks, dtype=torch.bool, device=self.device)
                    V = self.get_value(states_t, salesmen_masks_t)
                    V_list.append(V.cpu().numpy())

        if eval_mode:
            return env_info
        else:

            gae, returns = self.compute_gae_returns(r_lsit, V_list)

            state = np.concatenate(state_list, axis=0)
            act_logp = np.concatenate(act_logp_list, axis=0)
            act_ent = np.concatenate(act_ent_list, axis=0)
            V = np.concatenate(V_list[:-1],axis=0)
            act_mask = np.concatenate(act_mask_list, axis=0)
            act = np.concatenate(act_list, axis=0)

            gae = np.concatenate(gae, axis=0)
            returns = np.concatenate(returns, axis=0)

            return (
                batch_graph,
                state,
                act_logp,
                act,
                act_ent,
                act_mask,
                gae,
                returns,
                V,
                expand_step,
                env_info["costs"]
            )

    def compute_gae_returns(self, r_list, V_list):
        T = len(r_list)
        gae = 0
        adv = []
        returns = []
        for i in reversed(range(T)):
            delta = r_list[i] + V_list[i+1] - V_list[i]
            gae = delta + self._gamma * self._lambda * gae
            adv.append(gae)
            returns.append(gae + V_list[i])
        adv.reverse()
        returns.reverse()
        return adv, returns

    def eval_episode(self, env, batch_graph, agent_num, exploit_mode="sample", info=None):
        with torch.no_grad():
            eval_info = self.run_batch_episode(env, batch_graph, agent_num, eval_mode=True, exploit_mode=exploit_mode)
            cost = np.max(eval_info["costs"], axis=1)
            trajectory = eval_info["trajectories"]
            return cost, trajectory

    def state_dict(self):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            # "model_act_optim": self.act_optim.state_dict(),
            # "model_conf_optim": self.conf_optim.state_dict(),
            "model_optim": self.optim.state_dict(),
        }
        return checkpoint

    def load_state_dict(self, checkpoint):
        self.model.load_state_dict(checkpoint["model_state_dict"])
        # self.act_optim.load_state_dict(checkpoint["model_act_optim"])
        # self.conf_optim.load_state_dict(checkpoint["model_conf_optim"])
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
