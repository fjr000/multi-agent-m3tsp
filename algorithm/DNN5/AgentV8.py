import torch
import torch.nn as nn
from model.n4Model.model_v8 import Model
from algorithm.DNN5.AgentBase import AgentBase
from utils.TensorTools import _convert_tensor


class Agent(AgentBase):
    def __init__(self, args, config):
        super(Agent, self).__init__(args, config, Model)
        self.model.to(self.device)

    def save_model(self, id):
        filename = f"AgentV8_{id}"
        super(Agent, self)._save_model(self.args.model_dir, filename)

    def load_model(self, id):
        filename = f"AgentV8_{id}"
        super(Agent, self)._load_model(self.args.model_dir, filename)

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
        cur_pos = states[:,:,1].long()
        stay_up_logp = actions_dist.log_prob(cur_pos)
        stay_up_logp = torch.where(torch.isinf(stay_up_logp), 0, stay_up_logp)

        if agents_logits is not None:
            agents_dist = torch.distributions.Categorical(logits=agents_logits)
            agents_logp = agents_dist.log_prob(agents_logits.argmax(dim=-1))
            agents_logp = torch.where(agents_mask, agents_logp, 0)
            agt_entropy = torch.where(agents_mask, agents_dist.entropy(), 0)
        else:
            agents_logp = None
            agt_entropy = None

        return acts, acts_no_conflict, act_logp, agents_logp, stay_up_logp, actions_dist.entropy(), agt_entropy

    def predict(self, states_t, masks_t, info=None):
        self.model.train()

        actions, actions_no_conflict, act_logp, agents_logp, stay_up_logp, act_entropy, agt_entropy = self._get_action_logprob(
            states_t, masks_t,
            mode="sample", info=info, eval=False)
        return actions.cpu().numpy(), actions_no_conflict.cpu().numpy(), act_logp, agents_logp,stay_up_logp, act_entropy, agt_entropy

    def learn(self, act_logp, agents_logp,stay_up_likelihood, act_ent, agt_ent, costs):
        self.model.train()
        self.train_count += 1
        agt_ent_loss = torch.tensor([0], device=self.device)
        agents_loss = torch.tensor([0], device=self.device)
        if self.args.only_one_instance:
            act_loss, agents_loss = self.__get_loss_only_instance(act_logp, agents_logp, costs)
        else:
            act_loss, agents_loss = self._get_loss(act_logp, agents_logp, costs)
        act_ent_loss = act_ent
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

        stay_up_loss = stay_up_likelihood.mean()
        loss += 0.01 * stay_up_loss
        loss /= self.args.accumulation_steps
        loss.backward()

        pre_grad = 0
        if self.train_count % self.args.accumulation_steps == 0:
            pre_grad = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_max_norm)
            self.optim.step()
            self.optim.zero_grad()
        if agents_logp is None:
            return act_loss.item(), 0, act_ent_loss.item(), 0
        return act_loss.item(), agents_loss.item(),stay_up_loss.item(), act_ent_loss.item(), agt_ent_loss.item(), pre_grad

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

        self.reset_graph(graph, agent_num)
        act_logp_list = []
        agents_logp_list = []
        stay_up_logp_list = []
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
                acts, acts_no_conflict, act_logp, agents_logp, stay_up_logp, act_entropy, agt_entropy = self.predict(states_t,
                                                                                                       salesmen_masks_t,
                                                                                                       info)
                act_logp_list.append(act_logp.unsqueeze(-1))
                stay_up_logp_list.append(stay_up_logp.unsqueeze(-1))
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
            stay_up_logp = torch.cat(stay_up_logp_list, dim=-1)
            stay_up_likelihood = torch.sum(stay_up_logp, dim=-1)

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
                stay_up_likelihood,
                act_ent,
                agt_ent,
                env_info["costs"]
            )