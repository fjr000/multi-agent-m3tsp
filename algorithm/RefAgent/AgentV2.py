import argparse
import torch
import torch.nn as nn
from utils.TensorTools import _convert_tensor
import numpy as np
from model.RefModel.Model import Model
from algorithm.RefAgent.AgentBase import AgentBase


class AgentV2(AgentBase):
    def __init__(self, args, config):
        super(AgentV2, self).__init__(args, config, Model)
        self.model.to(self.device)
        self._gamma = 1

    def save_model(self, id):
        filename = f"RefAgentV2_{id}"
        super(AgentV2, self)._save_model(self.args.model_dir, filename)

    def load_model(self, id):
        filename = f"RefAgentV2_{id}"
        super(AgentV2, self)._load_model(self.args.model_dir, filename)

    def get_cumulative_returns_batch(self, np_batch_multi_reward):
        return_numpy = np.zeros_like(np_batch_multi_reward)
        return_numpy[...,-1] = np_batch_multi_reward[...,-1]
        for idx in range(-2, -np_batch_multi_reward.shape[-1] - 1, -1):
            return_numpy[...,idx] = np_batch_multi_reward[..., idx] + self._gamma * return_numpy[..., idx + 1]
        return return_numpy
    def run_batch_episode(self, env, batch_graph, agent_num, eval_mode=False, exploit_mode="sample", info=None):
        states, env_info = env.reset(
            config={
                "cities": batch_graph.shape[1],
                "salesmen": agent_num,
                "mode": "fixed",
                "N_aug": batch_graph.shape[0],
                "use_conflict_model": info.get("use_conflict_model", False) if info is not None else False,
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
        agt_reward_list = []

        done = False
        use_conflict_model = False
        while not done:
            states_t = _convert_tensor(states, device=self.device)
            # mask: true :not allow  false:allow

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
            agt_reward_list.append(r[...,None])
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
            # act_likelihood = torch.sum(act_logp, dim=-1)

            reward = np.concatenate(agt_reward_list, axis=-1)
            returns = self.get_cumulative_returns_batch(reward)


            if use_conflict_model:
                agents_logp = torch.cat(agents_logp_list, dim=-1)
                agt_ent = torch.cat(agt_ent_list, dim=-1)
                agt_ent = agt_ent.sum() / agt_ent.count_nonzero()
                # agents_logp = torch.where(agents_logp == 0, 0.0, agents_logp)
                # agents_likelihood = torch.sum(agents_logp, dim=-1)
            else:
                agents_logp = None
                agents_likelihood = None
                agt_ent = None

            # act_likelihood = torch.sum(act_logp, dim=-1) / act_logp.count_nonzero(dim = -1)
            # agents_likelihood = torch.sum(agents_logp, dim=-1) / agents_logp.count_nonzero(dim=-1)
            return (
                act_logp,
                agents_logp,
                act_ent,
                agt_ent,
                env_info["costs"],
                returns
            )

    def _get_loss(self, act_logp, agents_logp, costs, returns):

        # # 智能体间平均， 组间最小化最大
        costs_8 = costs.reshape(costs.shape[0] // 8, 8, -1)  # 将成本按实例组进行分组
        act_logp_8 = act_logp.reshape(act_logp.shape[0] // 8, 8,act_logp.size(1), act_logp.size(2))  # 将动作概率按实例组进行分组
        agt_logp_8 = agents_logp.reshape(agents_logp.size(0) // 8,8 , agents_logp.size(1), agents_logp.size(2))
        # #
        agents_max_cost = np.max(costs_8, keepdims=True, axis=-1)
        # # # # 智能体间优势
        group_adv = (agents_max_cost - np.mean(agents_max_cost, keepdims=True, axis=1)) / (
                agents_max_cost.std(keepdims=True, axis=1) + 1e-8)
        group_adv_t = _convert_tensor(group_adv, device=self.device)
        act_likelihood = torch.sum(act_logp_8, dim = -1)
        agt_likelihood = torch.sum(agt_logp_8, dim = -1)
        act_loss = (act_likelihood * group_adv_t).mean()
        agt_loss = (agt_likelihood * group_adv_t).mean()

        # returns_8 = returns.reshape(returns.shape[0] // 8, 8, returns.shape[1], -1)
        # returns_t = _convert_tensor(returns_8, device=self.device)
        # returns_t_b = (returns_t - returns_t.mean()) / (
        #     returns_t.std() + 1e-8
        # )
        # act_loss = - (act_logp_8 * returns_t).mean()
        # agt_loss = - (agt_logp_8 * returns_t).mean()

        return act_loss, agt_loss

    def _get_loss_only_instance(self, act_logp, agents_logp, costs, returns):
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

    def learn(self, act_logp, agents_logp, act_ent, agt_ent, costs,returns):
        self.model.train()

        loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        agt_ent_loss = torch.tensor([0], device=self.device)
        agents_loss = torch.tensor([0], device=self.device)

        if self.args.only_one_instance:
            act_loss, agents_loss = self._get_loss_only_instance(act_logp, agents_logp, costs, returns)
        else:
            act_loss, agents_loss = self._get_loss(act_logp, agents_logp, costs, returns)
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
            act_loss = torch.tensor([0], device=self.device)
            act_ent_loss = torch.tensor([0], device=self.device)

        if not torch.isnan(agt_ent_loss) and not torch.isclose(loss, torch.zeros((1,), device=self.device)):
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
        return act_loss.item(), agents_loss.item(), act_ent_loss.item(), agt_ent_loss.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_num", type=int, default=5)
    parser.add_argument("--agent_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_max_norm", type=float, default=1.0)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--returns_norm", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=float, default=256)
    parser.add_argument("--model_dir", type=str, default="../../pth/")
    args = parser.parse_args()

    from envs.GraphGenerator import GraphGenerator as GG

    graphG = GG(args.batch_size, 50, 2)
    graph = graphG.generate()
    from envs.MTSP.MTSP4 import MTSPEnv

    env = MTSPEnv()
    from algorithm.OR_Tools.mtsp import ortools_solve_mtsp

    # indexs, cost, used_time = ortools_solve_mtsp(graph, args.agent_num, 10000)
    # env.draw(graph, cost, indexs, used_time, agent_name="or_tools")
    # print(f"or tools :{cost}")
    from model.n4Model.config import Config

    agent = AgentV1(args, Config)
    min_greedy_cost = 1000
    min_sample_cost = 1000
    loss_list = []
    act_logp, agents_logp, costs = agent.run_batch_episode(env, graph, args.agent_num, eval_mode=False,
                                                           exploit_mode="sample")

    act_loss, agents_loss = agent.learn(act_logp, agents_logp, costs)