import torch
import torch.nn as nn
import torch.optim as optim
from utils.TensorTools import _convert_tensor
import numpy as np
import random
import os

class AgentBase:
    def __init__(self, args, config, model_class):
        self.args = args
        self.model = model_class(config)
        torch.set_float32_matmul_precision('high')
        # self.model = torch.compile(self.model, mode='default')
        self.conflict_model = None

        self.lr = args.lr
        self.grad_max_norm = args.grad_max_norm
        self.optim = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, "min",
                                                                       patience=10000 / args.eval_interval, factor=0.95,
                                                                       min_lr=1e-5)
        self.device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() and self.args.use_gpu else "cpu")
        self.model.to(self.device)
        self.train_count = 0

    def reset_graph(self, graph, n_agents, repeat_times = 1):
        """
        :param graph: [B,N,2]
        :return:
        """
        graph_t = _convert_tensor(graph, device=self.device, target_shape_dim=3)
        self.model.init_city(graph_t, n_agents, repeat_times)

    def _get_action_logprob(self, states, masks, mode="greedy", info=None, eval=False):
        info = {} if info is None else info
        info.update({
            "mode": mode,
        })
        actions_logits, acts = self.model(states, masks, info, eval=eval)
        if eval:
            return acts, None, None
        actions_dist = torch.distributions.Categorical(logits=actions_logits)
        act_logp = actions_dist.log_prob(acts)

        return acts, act_logp, actions_dist.entropy()

    def predict(self, states_t, masks_t, info=None):
        self.model.train()

        actions, act_logp,act_entropy = self._get_action_logprob(
            states_t, masks_t,
            mode="sample", info=info, eval=False)
        return actions.cpu().numpy(), act_logp, act_entropy

    def exploit(self, states_t, masks_t, mode="greedy", info=None):
        self.model.eval()
        actions, _, _ = self._get_action_logprob(states_t, masks_t, mode=mode, info=info, eval=True)
        return actions.cpu().numpy()

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


    def _get_loss(self, act_logp, costs):

        # 智能体间平均， 组间最小化最大
        costs_8 = costs.reshape(costs.shape[0] // 8, 8, -1)  # 将成本按实例组进行分组
        act_logp_8 = act_logp.reshape(act_logp.shape[0] // 8, 8, -1)  # 将动作概率按实例组进行分组

        agents_max_cost = np.max(costs_8, axis=-1)
        group_adv = (agents_max_cost - np.mean(agents_max_cost, keepdims=True, axis=1)) / (
                agents_max_cost.std(keepdims=True, axis=1) + 1e-8)
        adv = group_adv

        # 转换为tensor并放到指定的device上
        adv_t = _convert_tensor(adv, device=self.device)

        # 对动作概率为零的样本进行掩码
        act_logp_8_sum = act_logp_8.sum(dim=-1)
        mask_ = ((act_logp_8_sum != 0) & (~torch.isnan(act_logp_8_sum)))

        # 计算动作网络的损失，mask之后加权平均
        act_loss = (torch.exp(act_logp_8_sum[mask_] - act_logp_8_sum[mask_].detach()) * adv_t[mask_]).mean()
        return act_loss

    def learn(self, act_logp, act_ent, costs):
        self.model.train()
        self.train_count += 1
        act_loss = self._get_loss(act_logp, costs)
        act_ent_loss = act_ent

        loss = act_loss + self.args.entropy_coef * (- act_ent_loss)
        loss /= self.args.accumulation_steps
        loss.backward()

        pre_grad = 0

        if self.train_count % self.args.accumulation_steps == 0:
            pre_grad = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_max_norm)
            self.optim.step()
            self.optim.zero_grad()

        def check(value):
            return None if value is None else (value.item() if isinstance(value, torch.Tensor) else value)

        return_info={
            "act_loss": check(act_loss),
            "act_ent_loss": check(act_ent_loss),
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
            dones_t = _convert_tensor(dones,dtype=torch.bool, device=self.device) if dones is not None else None
            info.update({
                "masks_in_salesmen": masks_in_salesmen_t,
                "mask": city_mask_t,
                "dones": dones_t
            })
            if eval_mode:
                acts = self.exploit(states_t, salesmen_masks_t, exploit_mode, info)
            else:
                acts, act_logp, act_entropy = self.predict(states_t, salesmen_masks_t, info)
                act_logp_list.append(act_logp.unsqueeze(-1))
                act_ent_list.append(act_entropy.unsqueeze(-1))
            states, r, done, env_info = env.step(acts + 1)
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

            return (
                act_likelihood,
                act_ent,
                env_info["costs"]
            )

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

    def state_dict(self, info = None):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_optim": self.optim.state_dict(),
            "model_scheduler":self.lr_scheduler.state_dict(),
            "rng_state":{
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state(self.device) if torch.cuda.is_available() else None,
                'torch_cuda_all':torch.cuda.get_rng_state_all()if torch.cuda.is_available() else None
            },
            "info": info,
        }
        return checkpoint

    def load_state_dict(self, checkpoint):
        if 'actions_model.agent_decoder.action.glimpse_K' in checkpoint['model_state_dict']:
            checkpoint['model_state_dict'].pop('actions_model.agent_decoder.action.glimpse_K')
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optim.load_state_dict(checkpoint["model_optim"])

        model_scheduler = checkpoint.get("model_scheduler", None)
        if model_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint["model_scheduler"])

        rng_state = checkpoint.get("rng_state", None)
        if rng_state is not None:
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch"].cpu())
            torch.cuda.set_rng_state(rng_state["torch_cuda"].cpu(), self.device)
            if "torch_cuda_all" in rng_state:
                torch.cuda.set_rng_state_all([x.cpu() for x in rng_state["torch_cuda_all"]])

        info = checkpoint.get("info", None)
        return info

    def _save_model(self, model_dir, filename, info = None):
        save_path = f"{model_dir}{filename}.pth"
        checkpoint = self.state_dict(info)
        torch.save(checkpoint, save_path)
        print(f"save {save_path} successfully")

    def _load_model(self, model_dir, filename):
        load_path = f"{model_dir}{filename}.pth"
        info = None
        if os.path.isfile(load_path):
            checkpoint = torch.load(load_path, weights_only=False, map_location=self.device)
            info = self.load_state_dict(checkpoint)
            print(f"load {load_path} successfully")
        else:
            print("model file doesn't exist")

        return info