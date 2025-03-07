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

        # self._gamma = args.gamma
        self.lr = args.lr
        self.grad_max_norm = args.grad_max_norm
        self.act_optim = optim.AdamW(self.model.actions_model.parameters(), lr=self.lr)
        self.conf_optim = optim.AdamW(self.model.conflict_model.parameters(), lr=self.lr)
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
        actions, actions_no_conflict, act_logp, agents_logp = self.__get_action_logprob(states_t, masks_t,
                                                                                        mode="sample")
        return actions.cpu().numpy(), actions_no_conflict.cpu().numpy(), act_logp, agents_logp

    def exploit(self, states_t, masks_t, mode="greedy"):
        self.model.eval()
        actions, actions_no_conflict, _, _ = self.__get_action_logprob(states_t, masks_t, mode=mode)
        return actions.cpu().numpy(), actions_no_conflict.cpu().numpy()

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

    def __get_loss(self, act_logp, agents_logp, costs):
        costs_8 = costs.reshape(costs.shape[0] // 8, 8, -1)
        act_logp_8 = act_logp.reshape(act_logp.shape[0] // 8, 8, -1)
        agents_logp_8 = agents_logp.reshape(agents_logp.shape[0] // 8, 8, -1)
        max_costs = -np.max(costs_8,keepdims=True, axis=-1)
        max_costs = max_costs.mean(keepdims=True, axis=1)
        adv = -costs_8 - max_costs
        adv = _convert_tensor(adv, device=self.device)
        act_loss = - (act_logp_8 * adv).mean()
        agents_loss = - (agents_logp_8 * adv).mean()
        return act_loss, agents_loss

    def learn(self, act_logp, agents_logp, costs):
        self.model.train()
        act_loss, conflict_loss = self.__get_loss(act_logp, agents_logp, costs)
        self.__update_net(self.act_optim, self.model.actions_model.parameters(), act_loss)
        self.__update_net(self.conf_optim, self.model.conflict_model.parameters(), conflict_loss)
        # del states_tb, actions_tb, returns_tb, masks_tb
        return act_loss.item(), conflict_loss.item()

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
        done = False
        info = None
        while not done:
            states_t = _convert_tensor(states, device=self.device)
            salesmen_masks_t = _convert_tensor(salesmen_masks, device=self.device)
            if eval_mode:
                acts, acts_no_conflict = self.exploit(states_t, salesmen_masks_t, exploit_mode)
            else:
                acts, acts_no_conflict, act_logp, agents_logp = self.predict(states_t, salesmen_masks_t)
                act_logp_list.append(act_logp.unsqueeze(-1))
                agents_logp_list.append(agents_logp.unsqueeze(-1))
            states, r, done, info = env.step(acts_no_conflict + 1)
            salesmen_masks = info["salesmen_masks"]

        if eval_mode:
            return info
        else:
            act_looklihood = torch.sum(torch.cat(act_logp_list, dim=-1), dim=-1)
            agents_looklihood = torch.sum(torch.cat(agents_logp_list, dim=-1), dim=-1)
            return (
                act_looklihood,
                agents_looklihood,
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
        }
        return checkpoint

    def load_state_dict(self, checkpoint):
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.act_optim.load_state_dict(checkpoint["model_act_optim"])
        self.conf_optim.load_state_dict(checkpoint["model_conf_optim"])

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
