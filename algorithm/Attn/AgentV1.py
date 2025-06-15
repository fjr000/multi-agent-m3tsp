from torch import nn

from model.AttnModel.Model import Model, Config
from algorithm.Attn.AgentBase import AgentBase
import argparse
import torch
from utils.TensorTools import _convert_tensor
from torch.amp import autocast, GradScaler    # ✅
torch.set_float32_matmul_precision('high')
import  numpy as np

class Agent(AgentBase):
    def __init__(self, args, config):
        super(Agent, self).__init__(args, config, Model)
        self.model.to(self.device)
        self.name = "Attn_AgentV1"
        self.scaler = GradScaler( enabled=True)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, "min",
                                                                       patience=4000 / args.eval_interval, factor=0.95,
                                                                       min_lr=1e-5)

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

        return acts, act_logp, acts_ncf, actions_dist.entropy()

    def predict(self, states_t, masks_t, info=None):
        self.model.train()

        actions, act_logp, acts_ncf, ent = self._get_action_logprob(
            states_t, masks_t,
            mode="sample", info=info, eval=False)
        return actions.cpu().numpy(), act_logp, acts_ncf.cpu().numpy() if acts_ncf is not None else None, ent

    def exploit(self, states_t, masks_t, mode="greedy", info=None):
        self.model.eval()
        actions, acts_ncf = self._get_action_logprob(states_t, masks_t, mode=mode, info=info, eval=True)
        return actions.cpu().numpy(), acts_ncf.cpu().numpy() if acts_ncf is not None else None

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
        with autocast("cuda",dtype=torch.bfloat16):  # <-- AMP 开始
            self.reset_graph(graph, agent_num)
        act_logp_list = []
        act_ent_list = []
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
            dones_t = _convert_tensor(dones, dtype=torch.bool, device=self.device) if dones is not None else None
            info.update({
                "masks_in_salesmen": masks_in_salesmen_t,
                "mask": city_mask_t,
                "dones": dones_t
            })

            with autocast("cuda", dtype=torch.bfloat16):  # <-- AMP 开始
                if eval_mode:
                    acts, act_nf = self.exploit(states_t, salesmen_masks_t, exploit_mode, info)
                else:
                    acts, act_logp, act_nf, act_entropy = self.predict(states_t, salesmen_masks_t, info)
                    act_logp_list.append(act_logp.unsqueeze(-1))
                    act_ent_list.append(act_entropy.unsqueeze(-1))
            if act_nf is not None:
                states, r, done, env_info = env.step(act_nf + 1)
            else:
                states, r, done, env_info = env.step(acts+1)
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
            # act_likelihood = torch.sum(act_logp, dim=-1)
            adv = self.compute_advs(env_info['costs'], env_info['penalty'][...,1:1+len(act_logp_list)], env_info['conflict_count'])

            return (
                act_logp,
                act_ent,
                env_info["costs"],
                adv
            )

    def compute_advs(self, costs, penalty, conflict_count):
        agent_num = costs.shape[1]
        rewards = -costs
        penalty = -penalty

        # 智能体间平均， 组间最小化最大
        group_size= self.args.augment * self.args.repeat_times

        rewards_group = rewards.reshape(rewards.shape[0] // group_size, group_size, -1)  # 将成本按实例组进行分组
        agents_max_rewards = np.min(rewards_group, axis=-1)

        # instance_mean_8 = np.mean(agents_max_rewards, axis=1)
        # instance_std_8 = np.std(agents_max_rewards, axis=1)
        final_rewards = penalty
        final_rewards[...,-1] = agents_max_rewards.reshape(-1,1).repeat(agent_num,axis=1)
        for i in reversed(range(final_rewards.shape[-1]-1)):
            final_rewards[...,i] += final_rewards[..., i+1]
        final_rewards_group = final_rewards.reshape(final_rewards.shape[0] // group_size, group_size, agent_num, -1)

        group_mean = final_rewards_group.mean(axis=(1, 2), keepdims=True)
        group_std = final_rewards_group.std(axis=(1, 2), keepdims=True)

        adv = (final_rewards_group - group_mean) / (group_std + 1e-8)
        adv = adv.reshape(costs.shape[0], agent_num, -1)

        return adv

    def _get_loss(self, act_logp, adv):
        # 转换为tensor并放到指定的device上
        mask_ = ~torch.isclose(act_logp, torch.zeros_like(act_logp))
        adv_t = _convert_tensor(adv, device=self.device)
        act_loss = - (act_logp * adv_t)[mask_].mean()
        return act_loss

    def learn(self, act_logp, act_ent, costs, adv):
        self.model.train()
        self.train_count += 1
        with autocast("cuda",dtype=torch.bfloat16):  # <-- AMP 开始
            act_loss = self._get_loss(act_logp, adv)
            act_ent_loss = act_ent

            loss = act_loss + self.args.entropy_coef * (- act_ent_loss)
            loss /= self.args.accumulation_steps
        self.scaler.scale(loss).backward()  # ① 缩放梯度

        pre_grad = 0.0
        if self.train_count % self.args.accumulation_steps == 0:
            # ② 先反缩放，再 clip
            self.scaler.unscale_(self.optim)
            pre_grad = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_max_norm)
            # ③ 真正 step & 更新 scale
            self.scaler.step(self.optim)
            self.scaler.update()
            self.optim.zero_grad(set_to_none=True)

        def check(value):
            return None if value is None else (value.item() if isinstance(value, torch.Tensor) else value)

        return_info={
            "act_loss": check(act_loss),
            "act_ent_loss": check(act_ent_loss),
            "grad": check(pre_grad),
        }

        return return_info

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
    loss_s = agent.learn(*output)
