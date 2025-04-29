import torch

from model.n4Model.model_v7 import Model
from algorithm.DNN5.AgentBase import AgentBase
from utils.TensorTools import _convert_tensor
import  numpy as np
import torch.nn as nn
class Agent(AgentBase):
    def __init__(self, args, config):
        super(Agent, self).__init__(args, config, Model)
        self.model.to(self.device)

    def save_model(self, id):
        filename = f"AgentV7_{id}"
        super(Agent, self)._save_model(self.args.model_dir, filename)

    def load_model(self, id):
        filename = f"AgentV7_{id}"
        super(Agent, self)._load_model(self.args.model_dir, filename)

    def _get_action_logprob(self, states, masks, mode="greedy", info=None):
        info = {} if info is None else info
        info.update({
            "mode": mode,
        })
        actions_logits, agents_logits, acts, acts_no_conflict, agents_mask,alpha_logp, alpha = self.model(states, masks, info)

        # actions_logits = actions_logits / 1.1

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

        return acts, acts_no_conflict, act_logp , agents_logp, actions_dist.entropy(), agt_entropy, alpha_logp, alpha

    def predict(self, states_t, masks_t, info=None):
        self.model.train()
        actions, actions_no_conflict, act_logp, agents_logp, act_entropy, agt_entropy, alpha_logp, alpha = self._get_action_logprob(
            states_t, masks_t,
            mode="sample", info=info)
        return actions.cpu().numpy(), actions_no_conflict.cpu().numpy(), act_logp, agents_logp, act_entropy, agt_entropy, alpha_logp, alpha

    def exploit(self, states_t, masks_t, mode="greedy", info=None):
        self.model.eval()
        actions, actions_no_conflict, _, _, _, _, _,_ = self._get_action_logprob(states_t, masks_t, mode=mode, info=info)
        return actions.cpu().numpy(), actions_no_conflict.cpu().numpy()

    def _get_loss(self, act_logp, agents_logp, alpha_logp, costs):

        # 智能体间平均， 组间最小化最大
        costs_8 = costs.reshape(costs.shape[0] // 8, 8, -1)  # 将成本按实例组进行分组

        # agents_avg_cost = np.mean(costs_8, keepdims=True, axis=-1)
        agents_max_cost = np.max(costs_8, keepdims=True, axis=-1)
        # 智能体间优势
        agents_adv = costs_8 - agents_max_cost
        agents_adv = (agents_adv - agents_adv.mean(keepdims=True, axis=-1)) / (
                agents_adv.std(axis=-1, keepdims=True) + 1e-8)
        # 实例间优势
        group_adv = (agents_max_cost - np.mean(agents_max_cost, keepdims=True, axis=1)) / (
                agents_max_cost.std(keepdims=True, axis=1) + 1e-8)
        # 组合优势
        adv = self.args.agents_adv_rate * agents_adv + group_adv

        # 转换为tensor并放到指定的device上
        adv_t = _convert_tensor(adv, device=self.device)

        logp = act_logp
        if agents_logp is not None:
            logp += agents_logp
        if alpha_logp is not None:
            logp += alpha_logp[:,None] * 0.1
        logp_8 = logp.reshape(logp.shape[0] // 8, 8, -1)  # 将动作概率按实例组进行分组
        loss = (torch.exp(logp_8 - logp_8.detach()) * adv_t).mean()

        return loss
    def learn(self, act_logp, agents_logp, alpha_logp, act_ent, agt_ent, alpha, costs):
        self.model.train()
        self.train_count += 1
        agt_ent_loss = torch.tensor([0], device=self.device)
        agents_loss = torch.tensor([0], device=self.device)
        if self.args.only_one_instance:
            act_loss, agents_loss = self.__get_loss_only_instance(act_logp, agents_logp, costs)
        else:
            act_loss = self._get_loss(act_logp, agents_logp,alpha_logp, costs)

        loss = act_loss - alpha * 0.01
        loss /= self.args.accumulation_steps
        loss.backward()

        pre_grad = 0
        if self.train_count % self.args.accumulation_steps == 0:
            pre_grad = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_max_norm)
            self.optim.step()
            self.optim.zero_grad()
        return act_loss.item(), alpha.item(), pre_grad

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

        graph = env_info["graph"]

        self.reset_graph(graph, agent_num)
        act_logp_list = []
        agents_logp_list = []
        act_ent_list = []
        agt_ent_list = []
        alpha_list = []
        alpha_logp_list = []
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
            info.update({
                "masks_in_salesmen": masks_in_salesmen_t,
                "mask": city_mask_t
            })
            if eval_mode:
                acts, acts_no_conflict = self.exploit(states_t, salesmen_masks_t, exploit_mode, info)
            else:
                acts, acts_no_conflict, act_logp, agents_logp, act_entropy, agt_entropy,alpha_logp, alpha = self.predict(states_t,
                                                                                                       salesmen_masks_t,
                                                                                                       info)
                act_logp_list.append(act_logp.unsqueeze(-1))
                act_ent_list.append(act_entropy.unsqueeze(-1))
                alpha_list.append(alpha)
                alpha_logp_list.append(alpha_logp.unsqueeze(-1))
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

            alpha_logp = torch.cat(alpha_logp_list, dim=-1)
            alpha_logp_likelihood = torch.sum(alpha_logp, dim=-1)

            alpha = torch.cat(alpha_list, dim=-1).mean()

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
                alpha_logp_likelihood,
                act_ent,
                agt_ent,
                alpha,
                env_info["costs"]
            )