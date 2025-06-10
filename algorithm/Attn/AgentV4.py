import argparse
import torch
import torch.nn as nn
import numpy as np

from model.AttnModel.ModelV4 import Model, Config
from algorithm.Attn.AgentBase import AgentBase
from utils.TensorTools import _convert_tensor


class Agent(AgentBase):
    def __init__(self, args, config):
        super(Agent, self).__init__(args, config, Model)
        self.model.to(self.device)
        self.name = "Attn_AgentV4"

    def save_model(self, id, info = None):
        filename = f"{self.name}_{id}"
        super(Agent, self)._save_model(self.args.model_dir, filename, info)

    def load_model(self, id):
        filename = f"{self.name}_{id}"
        return super(Agent, self)._load_model(self.args.model_dir, filename)

    def _get_action_logprob(self, states, masks, mode="greedy", info=None, eval=False):
        info = {} if info is None else info
        info.update({
            "mode": mode,
        })
        actions_logits, acts, acts_ncf = self.model(states, masks, info, eval=eval)
        if eval:
            return acts, acts_ncf
        actions_dist = torch.distributions.Categorical(logits=actions_logits)
        act_logp = actions_dist.log_prob(acts)

        return acts, act_logp, acts_ncf

    def predict(self, states_t, masks_t, info=None):
        self.model.train()

        actions, act_logp, acts_ncf = self._get_action_logprob(
            states_t, masks_t,
            mode="sample", info=info, eval=False)
        return actions.cpu().numpy(), act_logp, acts_ncf.cpu().numpy() if acts_ncf is not None else None

    def exploit(self, states_t, masks_t, mode="greedy", info=None):
        self.model.eval()
        actions, acts_ncf = self._get_action_logprob(states_t, masks_t, mode=mode, info=info, eval=True)
        return actions.cpu().numpy(), acts_ncf.cpu().numpy() if acts_ncf is not None else None

    def _get_loss(self, new_act_logp, old_act_logp, adv):

        # 转换为tensor并放到指定的device上
        adv_t = _convert_tensor(adv, device=self.device)
        ratio = torch.exp(new_act_logp - old_act_logp)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.28) * (adv_t)
        act_loss = -torch.min(surr1, surr2).sum(-1).mean()
        return act_loss

    def learn(self, buffer):
        self.model.train()

        self.reset_graph(buffer['graph'], buffer['agent_num'], buffer['length'])
        logits, _, _ = self.model(_convert_tensor(buffer['states'],device=self.device),
                               _convert_tensor(buffer['salesmen_masks'], dtype=torch.bool, device = self.device),
                               {
                                    "mask": _convert_tensor(buffer['city_mask'], dtype=torch.bool, device=self.device),
                                    "dones": None,
                                }
                               )

        undones = ~buffer['dones']
        logits = logits[undones]
        dist = torch.distributions.Categorical(logits=logits)
        act = buffer['act'][undones]
        act_logp = buffer['act_logp'][undones]
        adv = buffer['adv'][undones]
        act_ent = dist.entropy().mean()

        new_act_logp = dist.log_prob(_convert_tensor( act,dtype=torch.long, device=self.device))
        old_act_logp = _convert_tensor(act_logp, device=self.device)

        self.train_count += 1
        act_loss = self._get_loss(new_act_logp, old_act_logp, adv)

        loss = act_loss
        loss /= self.args.accumulation_steps
        loss.backward()

        del dist, new_act_logp, old_act_logp, adv

        pre_grad = 0

        if self.train_count % self.args.accumulation_steps == 0:
            pre_grad = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_max_norm)
            self.optim.step()
            self.optim.zero_grad()

        def check(value):
            return None if value is None else (value.item() if isinstance(value, torch.Tensor) else value)

        return_info = {
            "act_loss": check(act_loss),
            "act_ent": check(act_ent),
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
        min_distance = env_info["min_distance"]

        self.reset_graph(graph, agent_num)
        act_logp_list = []
        info = {} if info is None else info

        states_list = []
        # masks_in_salesmen_list = []
        act_lsit = []
        salesmen_masks_list = []
        city_mask_list = []
        dones_list = []

        done = False
        while not done:
            states_t = _convert_tensor(states, device=self.device)
            # mask: true :not allow  false:allow

            salesmen_masks_t = _convert_tensor(~salesmen_masks, dtype=torch.bool, device=self.device)
            if masks_in_salesmen is not None:
                masks_in_salesmen_t = _convert_tensor(~masks_in_salesmen, dtype=torch.bool, device=self.device)
            else:
                masks_in_salesmen_t = None
            city_mask_t = _convert_tensor(~city_mask, dtype=torch.bool, device=self.device)
            dones_t = _convert_tensor(dones, dtype=torch.bool, device=self.device) if dones is not None else None
            info.update({
                "masks_in_salesmen": masks_in_salesmen_t,
                "mask": city_mask_t,
                "dones": dones_t
            })

            # 样本记录
            states_list.append(states.copy())
            # masks_in_salesmen_list.append(~masks_in_salesmen)
            salesmen_masks_list.append(~salesmen_masks)
            city_mask_list.append(~city_mask)
            if dones is None:
                dones_list.append(np.zeros(states.shape[0], dtype=np.bool_))
            else:
                dones_list.append(dones)

            if eval_mode:
                acts, acts_ncf = self.exploit(states_t, salesmen_masks_t, exploit_mode, info)
            else:
                acts, act_logp, acts_ncf = self.predict(states_t, salesmen_masks_t, info)
                act_logp_list.append(act_logp)
                act_lsit.append(acts)

            if acts_ncf is None:
                states, r, done, env_info = env.step(acts + 1)
            else:
                states, r, done, env_info = env.step(acts_ncf + 1)
            salesmen_masks = env_info["salesmen_masks"]
            masks_in_salesmen = env_info["masks_in_salesmen"]
            city_mask = env_info["mask"]
            dones = env_info["dones"]

        if eval_mode:
            return env_info
        else:
            act_logp = torch.cat(act_logp_list, dim=0)
            # act_logp = torch.where(act_logp == 0, 0.0, act_logp)  # logp为0时置为0

            buffer_states = np.concatenate(states_list, axis=0)
            buffer_salesmen = np.concatenate(salesmen_masks_list, axis=0)
            buffer_city_mask = np.concatenate(city_mask_list, axis=0)
            buffer_dones = np.concatenate(dones_list, axis=0)
            buffer_act_logp = act_logp.detach().cpu().numpy()
            buffer_act = np.concatenate(act_lsit, axis=0)

            adv = self.compute_advs(env_info['costs'], len(act_logp_list), agent_num / graph.shape[1] * min_distance, env_info['conflict_count'])

            # adv = self.compute_advs_with_stayup_penalty(env_info['costs'], min_distance, env_info['trajectories'])

            buffer = {
                "graph":graph,
                "agent_num": agent_num,

                "states": buffer_states,
                "salesmen_masks": buffer_salesmen,
                "city_mask": buffer_city_mask,
                "dones": buffer_dones,
                "act":buffer_act,
                "act_logp": buffer_act_logp,
                "adv": adv,
                "length": len(act_logp_list)
            }

            return buffer


    def compute_advs(self, costs, repeat_times, stand_penalty_meta = 0, stand_count = None):
        agent_num = costs.shape[1]
        if stand_count is None:
            rewards = -costs
        else:
            rewards = -costs - stand_count * stand_penalty_meta[:,np.newaxis]
        # 智能体间平均， 组间最小化最大
        rewards_8 = rewards.reshape(rewards.shape[0] // 8, 8, -1)  # 将成本按实例组进行分组
        agents_max_rewards = np.min(rewards_8, axis=-1)
        group_adv = (agents_max_rewards - np.mean(agents_max_rewards, keepdims=True, axis=1)) / (
                agents_max_rewards.std(keepdims=True, axis=1) + 1e-8)
        adv = group_adv.reshape(1, -1, 1)
        adv = adv.repeat(agent_num, axis=2).repeat(repeat_times, axis=0).reshape(-1, agent_num)
        return adv


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--agent_num", type=int, default=5)
    parser.add_argument("--fixed_agent_num", type=bool, default=False)
    parser.add_argument("--agent_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--grad_max_norm", type=float, default=0.5)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=1e-3)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--city_nums", type=int, default=50)
    parser.add_argument("--random_city_num", type=bool, default=False)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=155008)
    parser.add_argument("--env_masks_mode", type=int, default=7,
                        help="0 for only the min cost  not allow back depot; 1 for only the max cost allow back depot")
    parser.add_argument("--eval_interval", type=int, default=1000, help="eval  interval")
    parser.add_argument("--use_conflict_model", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--train_conflict_model", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--train_actions_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_city_encoder", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--use_agents_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--use_city_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--agents_adv_rate", type=float, default=0.0, help="rate of adv between agents")
    parser.add_argument("--conflict_loss_rate", type=float, default=1.0, help="rate of adv between agents")
    parser.add_argument("--only_one_instance", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--save_model_interval", type=int, default=5000, help="save model interval")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--epoch_size", type=int, default=1280000, help="number of instance for each epoch")
    parser.add_argument("--n_epoch", type=int, default=100, help="number of epoch")
    args = parser.parse_args()

    from envs.GraphGenerator import GraphGenerator as GG

    graphG = GG(args.batch_size, 50, 2)
    graph = graphG.generate()
    from envs.MTSP.MTSP5 import MTSPEnv

    env = MTSPEnv()
    agent = Agent(args, Config)
    min_greedy_cost = 1000
    min_sample_cost = 1000
    loss_list = []

    train_info={
        "use_conflict_model": args.use_conflict_model,
        "train_conflict_model":args.train_conflict_model,
        "train_actions_model": args.train_actions_model,
    }

    city_nums = args.city_nums
    agent_num = args.agent_num

    if args.only_one_instance:
        graph = graphG.generate(1).repeat(args.batch_size, axis=0)
        graph_8 = GG.augment_xy_data_by_8_fold_numpy(graph)
    else:
        graph = graphG.generate(args.batch_size, city_nums)
        graph_8 = GG.augment_xy_data_by_8_fold_numpy(graph)

    output = agent.run_batch_episode(env, graph_8, agent_num, eval_mode=False, info=train_info)
    loss_s = agent.learn(output)
