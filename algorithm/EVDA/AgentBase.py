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
        # self.act_optim = optim.AdamW(self.model.actions_model.parameters(), lr=self.lr)
        # self.conf_optim = optim.AdamW(self.model.conflict_model.parameters(), lr=self.lr)
        self.optim = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() and self.args.use_gpu else "cpu")
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, "min",
                                                                       patience=10000 / args.eval_interval, factor=0.95,
                                                                       min_lr=1e-5)
        self.model.to(self.device)
        self.train_count = 0

    def reset_graph(self, graph, n_agents):
        """
        :param graph: [B,N,2]
        :return:
        """
        graph_t = _convert_tensor(graph, device=self.device, target_shape_dim=3)
        value = self.model.init_city(graph_t, n_agents).squeeze(1)
        return value

    def _get_action_logprob(self, states, masks, mode="greedy", info=None, eval=False):
        info = {} if info is None else info
        info.update({
            "mode": mode,
        })
        actions_logits, agents_logits, acts, acts_no_conflict, agents_mask = self.model(states, masks, info, eval=eval)
        if eval:
            return acts, acts_no_conflict, None, None, None, None
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

        actions, actions_no_conflict, act_logp, agents_logp, act_entropy, agt_entropy = self._get_action_logprob(
            states_t, masks_t,
            mode="sample", info=info, eval=False)
        return actions.cpu().numpy(), actions_no_conflict.cpu().numpy(), act_logp, agents_logp, act_entropy, agt_entropy

    def exploit(self, states_t, masks_t, mode="greedy", info=None):
        self.model.eval()
        actions, actions_no_conflict, _, _, _, _ = self._get_action_logprob(states_t, masks_t, mode=mode, info=info, eval=True)
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

    def _compute_adv(self, costs, value):
        with torch.no_grad():
            costs_t = _convert_tensor(costs, device=self.device)
            adv = (costs_t - value.detach())

            adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)

        return adv_norm

    def _get_value_loss(self, instance_costs, value):
        instance_costs_t = _convert_tensor(instance_costs, device=self.device)
        value_loss = nn.MSELoss()(instance_costs_t, value)
        return value_loss

    def _get_policy_loss(self, act_logp, agents_logp, costs, value):

        adv = self._compute_adv(costs, value)

        act_loss = (act_logp * adv).sum(dim=-1).mean()
        if agents_logp is not None:
            agents_loss = (agents_logp * adv).sum(dim=-1).mean()
        else:
            agents_loss = None

        return act_loss, agents_loss

    def learn(self, act_logp, agents_logp, act_ent, agt_ent, costs, instance_costs, value):
        self.model.train()
        self.train_count += 1

        act_ent_loss = act_ent
        agt_ent_loss = None

        act_loss, agents_loss = self._get_policy_loss(act_logp, agents_logp, costs, value)
        value_loss = self._get_value_loss(instance_costs, value)
        loss = torch.zeros((1), device=self.device)

        if agents_logp is not None and self.args.train_conflict_model:
            # 修改为检查 agents_loss 是否包含 NaN
            if not torch.any(torch.isnan(agents_loss)):
                agt_ent_loss = agt_ent
            else:
                agents_loss = torch.tensor([0], device=self.device)
                agt_ent_loss = torch.tensor([0], device=self.device)
            # 更新损失计算，确保使用正确的变量名称
            loss += self.args.conflict_loss_rate * agents_loss + self.args.entropy_coef * (- agt_ent_loss)

        if self.args.train_actions_model:
            loss += act_loss + self.args.entropy_coef * (- act_ent_loss)

        loss += 0.5 * value_loss

        loss /= self.args.accumulation_steps
        loss.backward()

        pre_grad = 0
        if self.train_count % self.args.accumulation_steps == 0:
            pre_grad = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_max_norm)
            self.optim.step()
            self.optim.zero_grad()

        def check(value):
            return None if value is None else value.item()

        return_info={
            "act_loss": check(act_loss),
            "agents_loss": check(agents_loss),
            "act_ent_loss": check(act_ent_loss),
            "agt_ent_loss": check(agt_ent_loss),
            "value_loss": check(value_loss),
            "grad": check(pre_grad),
        }

        return return_info

    def run_batch_episode(self, env, batch_graph, agent_num, eval_mode=False, exploit_mode="sample", info=None):
        config = {
            "cities": batch_graph.shape[1],
            "salesmen": agent_num,
            "mode": "fixed",
            "N_aug": batch_graph.shape[0],
            "use_conflict_model": info.get("use_conflict_model", False) if info is not None else False,
        }
        if info is not None and info.get("trajs", None) is not None:
            config.update({
                "trajs": info["trajs"]
            })
        states, env_info = env.reset(
            config=config,
            graph=batch_graph
        )
        salesmen_masks = env_info["salesmen_masks"]
        masks_in_salesmen = env_info["masks_in_salesmen"]
        city_mask = env_info["mask"]
        dones = env_info["dones"]

        graph = env_info["graph"]

        value = self.reset_graph(graph, agent_num)
        act_logp_list = []
        agents_logp_list = []
        act_ent_list = []
        agt_ent_list = []
        info = {} if info is None else info

        done = False
        use_conflict_model = False
        while not done:
            states_t = _convert_tensor(states, device=self.device)
            # mask: true :not allow  false:allow

            salesmen_masks_t = _convert_tensor(~salesmen_masks, dtype=torch.bool, device=self.device)
            if masks_in_salesmen is not None:
                masks_in_salesmen_t = _convert_tensor(~masks_in_salesmen, dtype=torch.bool, device=self.device)
            else:
                masks_in_salesmen_t = None
            city_mask_t = _convert_tensor(~city_mask, dtype=torch.bool, device=self.device)
            dones_t = _convert_tensor(dones,dtype=torch.bool, device=self.device) if dones is not None else None
            info.update({
                "masks_in_salesmen": masks_in_salesmen_t,
                "mask": city_mask_t,
                "dones": dones_t
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
            dones = env_info["dones"]


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

            instance_cost = self._compute_instance_cost(env_info["costs"])

            return (
                act_likelihood,
                agents_likelihood,
                act_ent,
                agt_ent,
                env_info["costs"],
                instance_cost,
                value
            )

    def _compute_instance_cost(self, costs):
        costs_8 = costs.reshape(costs.shape[0] // 8, 8, -1)
        costs_8 = np.max(costs_8, axis=-1)
        instance_costs = np.mean(costs_8, keepdims=True, axis=-1).repeat(8,axis=-1).flatten()[:,np.newaxis]
        return instance_costs

    def eval_episode(self, env, batch_graph, agent_num, exploit_mode="sample", info=None):
        with torch.inference_mode():
            eval_info = self.run_batch_episode(env, batch_graph, agent_num, eval_mode=True, exploit_mode=exploit_mode,
                                               info=info)
            cost = np.max(eval_info["costs"], axis=1)
            trajectory = eval_info["trajectories"]
            return cost, trajectory

    def eval_tsp_episode(self, env, batch_graph, trajs, exploit_mode="sample", info=None):
        info = info if info is not None else {}
        info.update({
            "trajs": trajs
        })
        with torch.no_grad():
            eval_info = self.run_batch_episode(env, batch_graph, 1, eval_mode=True, exploit_mode=exploit_mode,
                                               info=info)
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
