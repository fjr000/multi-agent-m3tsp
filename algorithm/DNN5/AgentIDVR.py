import argparse
import time

from torch import nn

from model.n4Model.model_critic import Model
import torch
from utils.TensorTools import _convert_tensor
import numpy as np
from algorithm.DNN5.AgentBase import AgentBase
import torch.nn.functional as F

class AgentIDVR(AgentBase):
    def __init__(self, args, config):
        super(AgentIDVR, self).__init__(args, config, Model)
        self.model.to(self.device)

    def save_model(self, id):
        filename = f"AgentIDVR_{id}"
        super(AgentIDVR, self)._save_model(self.args.model_dir, filename)

    def load_model(self, id):
        filename = f"AgentIDVR_{id}"
        super(AgentIDVR, self)._load_model(self.args.model_dir, filename)

    def __get_action_logprob(self, states, masks, mode="greedy", info=None):
        info = {} if info is None else info
        info.update({
            "mode": mode,
        })
        actions_logits, agents_logits, acts, acts_no_conflict, agents_mask, V = self.model(states, masks, info)
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

        return acts, acts_no_conflict, act_logp, agents_logp, actions_dist.entropy(), agt_entropy, V

    def predict(self, states_t, masks_t, info=None):
        self.model.train()
        actions, actions_no_conflict, act_logp, agents_logp, act_entropy, agt_entropy, V = self.__get_action_logprob(
            states_t, masks_t,
            mode="sample", info=info)
        return actions.cpu().numpy(), actions_no_conflict.cpu().numpy(), act_logp, agents_logp, act_entropy, agt_entropy, V

    def exploit(self, states_t, masks_t, mode="greedy", info=None):
        self.model.eval()
        actions, actions_no_conflict, _, _, _, _, _ = self.__get_action_logprob(states_t, masks_t, mode=mode, info=info)
        return actions.cpu().numpy(), actions_no_conflict.cpu().numpy()

    def __get_logprob(self, states, masks, actions):
        actions_logits, agents_logits, acts, acts_no_conflict, _ = self.model(states, masks)
        dist = torch.distributions.Categorical(logits=actions_logits)
        agents_dist = torch.distributions.Categorical(logits=agents_logits)
        agents_logp = agents_dist.log_prob(agents_logits.argmax(dim=-1))
        entropy = dist.entropy()
        return dist.log_prob(actions), entropy, agents_logp

    def _get_gae(self, rewards, V):
        rewards_t = _convert_tensor(rewards, device=self.device)
        delta = rewards_t + V[...,1:]  - V[...,:-1]
        advantages = torch.zeros_like(rewards_t, device=self.device)
        gae = 0

        for i in reversed(range(rewards.shape[2])):
            gae = delta[...,i] + gae
            advantages[...,i] = gae
        returns = advantages + V[...,:-1]
        return advantages, returns

    def __get_value_loss (self, returns, V):

        loss = F.mse_loss(V[...,:-1], returns)
        return loss

    def _get_loss(self, act_logp, agents_logp, gae, costs):
        costs_8 = costs.reshape(costs.shape[0] // 8, 8, -1)
        max_cost_8 = np.max(costs_8, axis=-1, keepdims=True)
        adv = (max_cost_8 - max_cost_8.mean(axis = 1, keepdims = True)) / (max_cost_8.std(axis = 1, keepdims = True) + 1e-6)
        adv = - adv.reshape(-1,1,1)
        adv_t = _convert_tensor(adv, device=self.device)
        gae = (gae- gae.mean()) / (gae.std()+1e-8)
        act_loss = -( act_logp * (0.3 * gae + adv_t)).mean()
        agt_loss = None
        if agents_logp is not None:
            agt_loss = -(agents_logp * gae).mean()
        return act_loss, agt_loss

    def learn(self, act_logp, agents_logp, act_ent, agt_ent, costs, V, rewards):
        self.model.train()

        loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        agt_ent_loss = torch.tensor([0], device=self.device)
        agents_loss = torch.tensor([0], device=self.device)

        gae, returns = self._get_gae(rewards, V)

        act_loss, agents_loss = self._get_loss(act_logp, agents_logp, gae, costs)

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
        else:
            agents_loss = torch.tensor([0], device=self.device)
            agt_ent_loss = torch.tensor([0], device=self.device)
        if self.args.train_actions_model:
            loss += act_loss + self.args.entropy_coef * (- act_ent_loss)
        else:
            act_loss = torch.tensor([0], device=self.device)
            act_ent_loss = torch.tensor([0], device=self.device)

        value_loss = self.__get_value_loss(returns, V)

        if not torch.isnan(agt_ent_loss) and not torch.isclose(loss, torch.zeros((1,), device=self.device)):
            loss += 0.5 * value_loss
            loss /= self.args.accumulation_steps
            loss.backward()
            self.train_count += 1
        else:
            del agents_logp, agt_ent, act_logp, act_ent,
            print("empty")
            torch.cuda.empty_cache()


        if self.train_count % self.args.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_max_norm)
            self.optim.step()
            self.optim.zero_grad()

        if (agt_ent_loss.item() > 10):
            pass
        return act_loss.item(), agents_loss.item(), act_ent_loss.item(), agt_ent_loss.item(), value_loss.item()

    def get_cumulative_returns_batch(self, np_batch_multi_reward):
        return_numpy = np.zeros_like(np_batch_multi_reward)
        return_numpy[...,-1] = np_batch_multi_reward[...,-1]
        for idx in range(-2, -np_batch_multi_reward.shape[-1] - 1, -1):
            return_numpy[...,idx] = np_batch_multi_reward[..., idx] + self._gamma * return_numpy[..., idx + 1]
        return return_numpy

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
        V_list = []
        rewards_list = []

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

            info = {} if info is None else info
            info.update({
                "masks_in_salesmen":masks_in_salesmen_t,
                "mask":city_mask_t
            })
            if eval_mode:
                acts, acts_no_conflict = self.exploit(states_t, salesmen_masks_t, exploit_mode, info)
            else:
                acts, acts_no_conflict, act_logp, agents_logp, act_entropy, agt_entropy, V = self.predict(states_t,
                                                                                                       salesmen_masks_t,
                                                                                                       info)
                act_logp_list.append(act_logp.unsqueeze(-1))
                act_ent_list.append(act_entropy.unsqueeze(-1))
                V_list.append(V)
                if agents_logp is not None:
                    use_conflict_model = True
                    agents_logp_list.append(agents_logp.unsqueeze(-1))
                    agt_ent_list.append(agt_entropy.unsqueeze(-1))
            states, r, done, env_info = env.step(acts_no_conflict + 1)

            rewards_list.append(r[...,None])
            salesmen_masks = env_info["salesmen_masks"]
            masks_in_salesmen = env_info["masks_in_salesmen"]
            city_mask = env_info["mask"]

            if done:
                states_t = _convert_tensor(states, device=self.device)
                salesmen_masks_t = _convert_tensor(~salesmen_masks, dtype=torch.bool, device=self.device)
                if self.args.use_agents_mask:
                    masks_in_salesmen_t = _convert_tensor(~masks_in_salesmen, dtype=torch.bool, device=self.device)
                else:
                    masks_in_salesmen_t = None

                if self.args.use_city_mask:
                    city_mask_t = _convert_tensor(~city_mask, dtype=torch.bool, device=self.device)
                else:
                    city_mask_t = None

                info = {} if info is None else info
                info.update({
                    "masks_in_salesmen": masks_in_salesmen_t,
                    "mask": city_mask_t
                })
                _, _, _, _, _, V = self.model(states_t, salesmen_masks_t, info)
                V_list.append(V)


        if eval_mode:
            return env_info
        else:
            act_logp = torch.cat(act_logp_list, dim=-1)
            act_ent = torch.cat(act_ent_list, dim=-1).mean()
            act_ent = act_ent.sum() / act_ent.count_nonzero()
            # act_logp = torch.where(act_logp == 0, 0.0, act_logp)  # logp为0时置为0
            # act_likelihood = torch.sum(act_logp, dim=-1)

            if use_conflict_model:
                agents_logp = torch.cat(agents_logp_list, dim=-1)
                agt_ent = torch.cat(agt_ent_list, dim=-1)
                agt_ent = agt_ent.sum() / agt_ent.count_nonzero()
                # agents_logp = torch.where(agents_logp == 0, 0.0, agents_logp)
                # agents_likelihood = torch.sum(agents_logp, dim=-1)
            else:
                agents_likelihood = None
                agt_ent = None

            # act_likelihood = torch.sum(act_logp, dim=-1) / act_logp.count_nonzero(dim = -1)
            # agents_likelihood = torch.sum(agents_logp, dim=-1) / agents_logp.count_nonzero(dim=-1)
            V_t = torch.cat(V_list, dim=-1)
            rewards_np = np.concatenate(rewards_list, axis=-1)
            return (
                act_logp,
                agents_logp,
                act_ent,
                agt_ent,
                env_info["costs"],
                V_t,
                rewards_np
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--agent_num", type=int, default=10)
    parser.add_argument("--fixed_agent_num", type=bool, default=False)
    parser.add_argument("--agent_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad_max_norm", type=float, default=1)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=1e-2)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--city_nums", type=int, default=50)
    parser.add_argument("--random_city_num", type=bool, default=True)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=0)
    parser.add_argument("--env_masks_mode", type=int, default=4,
                        help="0 for only the min cost  not allow back depot; 1 for only the max cost allow back depot")
    parser.add_argument("--eval_interval", type=int, default=400, help="eval  interval")
    parser.add_argument("--use_conflict_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_conflict_model", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--train_actions_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_city_encoder", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--use_agents_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--use_city_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--agents_adv_rate", type=float, default=0.1, help="rate of adv between agents")
    parser.add_argument("--conflict_loss_rate", type=float, default=0.1, help="rate of adv between agents")
    parser.add_argument("--only_one_instance", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--save_model_interval", type=int, default=10000, help="save model interval")
    parser.add_argument("--seed", type=int, default=528, help="random seed")
    args = parser.parse_args()

    from envs.GraphGenerator import GraphGenerator as GG

    graphG = GG(args.batch_size, 50, 2)
    graph = graphG.generate()
    from envs.MTSP.MTSP5_IDVR import MTSPEnv_IDVR as MTSPEnv

    env = MTSPEnv({
        "env_masks_mode": args.env_masks_mode,
        "use_conflict_model":args.use_conflict_model
    })
    from algorithm.OR_Tools.mtsp import ortools_solve_mtsp

    # indexs, cost, used_time = ortools_solve_mtsp(graph, args.agent_num, 10000)
    # env.draw(graph, cost, indexs, used_time, agent_name="or_tools")
    # print(f"or tools :{cost}")
    from model.n4Model.config import Config

    agent = AgentIDVR(args, Config)
    min_greedy_cost = 1000
    min_sample_cost = 1000
    loss_list = []
    inp = agent.run_batch_episode(env, graph, args.agent_num, eval_mode=False,
                                                           exploit_mode="sample")

    out = agent.learn(*inp)