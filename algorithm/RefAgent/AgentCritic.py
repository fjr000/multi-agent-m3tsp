import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
from model.RefModel.ModelCritic import Model
from algorithm.RefAgent.AgentBase import AgentBase
import numpy as np
from utils.TensorTools import _convert_tensor

class AgentCritic(AgentBase):
    def __init__(self, args, config):
        super(AgentCritic, self).__init__(args, config, Model)
        self.model.to(self.device)

    def save_model(self, id):
        filename = f"RefAgentCritic_{id}"
        super(AgentCritic, self)._save_model(self.args.model_dir, filename)

    def load_model(self, id):
        filename = f"RefAgentCritic_{id}"
        super(AgentCritic, self)._load_model(self.args.model_dir, filename)

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

    def _get_gae(self, costs, V):
        td_error = V[...,1:] - V[...,:-1]
        td_error[...,-1] += _convert_tensor(-costs, device=self.device)
        gae = torch.cumsum(td_error.flip(-1), dim=-1).flip(-1)
        return gae

    def __get_value_loss (self, costs, V):

        loss = F.mse_loss(V, _convert_tensor(-costs[:,:,None].repeat(V.size(2), axis = 2), device=self.device))
        return loss

    def _get_loss(self, act_logp, agents_logp, gae):
        act_loss = -( act_logp[...,:-1] * gae).mean()
        agt_loss = None
        if agents_logp is not None:
            agt_loss = -(agents_logp[...,:-1] * gae).mean()
        return act_loss, agt_loss

    def learn(self, act_logp, agents_logp, act_ent, agt_ent, costs, V):
        self.model.train()

        loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        agt_ent_loss = torch.tensor([0], device=self.device)
        agents_loss = torch.tensor([0], device=self.device)

        gae = self._get_gae(costs, V)

        act_loss, agents_loss = self._get_loss(act_logp, agents_logp, gae)

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

        value_loss = self.__get_value_loss(costs, V)

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
        V_list = []

        done = False
        use_conflict_model = False
        while not done:
            states_t = _convert_tensor(states, device=self.device)
            # mask: true :not allow  false:allow

            salesmen_masks_t = _convert_tensor(~salesmen_masks, dtype= torch.bool, device=self.device)
            if self.args.use_agents_mask:
                masks_in_salesmen_t = _convert_tensor(~masks_in_salesmen, dtype= torch.bool, device=self.device)
            else:
                masks_in_salesmen_t = None

            if self.args.use_city_mask:
                city_mask_t = _convert_tensor(~city_mask, dtype= torch.bool, device=self.device)
            else:
                city_mask_t = None

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
            return (
                act_logp,
                agents_logp,
                act_ent,
                agt_ent,
                env_info["costs"],
                V_t
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

    agent = AgentCritic(args, Config)
    min_greedy_cost = 1000
    min_sample_cost = 1000
    loss_list = []
    act_logp, agents_logp, costs = agent.run_batch_episode(env, graph, args.agent_num, eval_mode=False,
                                                           exploit_mode="sample")

    act_loss, agents_loss = agent.learn(act_logp, agents_logp, costs)