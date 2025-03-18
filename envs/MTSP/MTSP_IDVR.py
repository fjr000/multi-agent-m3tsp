import argparse
import copy

from envs.MTSP.MTSP4 import MTSPEnv
import numpy as np
class MTSPEnv_IDVR(MTSPEnv):
    def __init__(self, config):
        super(MTSPEnv_IDVR, self).__init__(config)
        self.last_costs = None

    def _init(self, graph=None):
        super(MTSPEnv_IDVR,self)._init(graph)
        self.last_costs = copy.deepcopy(self.costs)


    def _get_reward(self):
        self.dones = np.all(self.stage_2, axis=1)
        self.done = np.all(self.dones, axis=0)
        # self.rewards = np.where(self.dones[:,None], -np.max(self.costs,keepdims=True, axis=1).repeat(self.salesmen, axis = 1), 0)
        # self.rewards = self.last_costs - self.costs
        # self.last_costs = copy.deepcopy(self.costs)
        self.rewards = -np.max(self.costs, keepdims=True, axis=1).repeat(self.salesmen, axis=1)
        return self.rewards

    def _get_salesmen_masks(self):
        B, N = self.mask.shape
        A = self.salesmen

        # 初始化批量掩码 [B, A, N]
        repeat_masks = np.tile(self.mask[:, None, :], (1, A, 1))
        # repeat_masks = self.mask[:,None,:].repeat(A,axis = 1)

        # # 处理停留限制 (使用位运算加速)
        # cur_pos = self.trajectories[..., self.step_count]-1
        # stay_update_mask = (self.remain_stay_still_log < self.stay_still_limit) & (cur_pos != 0)
        # repeat_masks[stay_update_mask, cur_pos[stay_update_mask] ] = 1

        active_agents = self.traj_stages == 1
        batch_indices_1d = self.batch_ar[np.sum(active_agents, axis=-1) > 1]
        batch_indices = batch_indices_1d[:,None]

        if self.env_masks_mode == 0:
            # 返回仓库优化 (新增阶段0排除)
            masked_cost = np.where(active_agents, self.costs, np.inf)
            masked_cost_sl = masked_cost[batch_indices_1d]
            min_cost_idx =  np.argmin(masked_cost_sl, axis=-1)
            # repeat_masks[batch_indices, :, 0] = 1
            # repeat_masks[batch_indices, min_cost_idx[:,None], 0] = 0
            # # 仅不允许最小开销的智能体留在原地
            # 允许所有激活智能体留在原地
            # 筛选有效批次的 mask 和 indices
            valid_mask = active_agents[batch_indices_1d]  # 形状 (K, A)
            valid_indices = self.cur_pos[batch_indices_1d]  # 形状 (K, A)

            # 使用高级索引直接赋值
            # x = repeat_masks[batch_indices, np.arange(A), valid_indices]
            repeat_masks[batch_indices, np.arange(A), valid_indices] = valid_mask
            # xx =  repeat_masks[batch_indices, np.arange(A), valid_indices]
            x_min_cur_pos = self.cur_pos[batch_indices_1d, min_cost_idx]
            repeat_masks[batch_indices, min_cost_idx[:,None], x_min_cur_pos[:,None]] = 0

            # for b_idx, min_idx in zip(batch_indices.squeeze(1), min_cost_idx):
            #     active_agts = active_agents[b_idx]
            #     for agent_idx in range(active_agents.shape[1]):
            #         if active_agts[agent_idx] and agent_idx != min_idx:
            #             pos = self.cur_pos[b_idx, agent_idx]
            #             repeat_masks[b_idx, agent_idx, pos] = 1
            #         else:
            #             pos = self.cur_pos[b_idx, agent_idx]
            #             repeat_masks[b_idx, agent_idx, pos] = 0
            # cur_pos = self.trajectories[..., self.step_count]-1
            # x_cur = cur_pos[batch_indices.squeeze(1), max_cost_idx]
            # repeat_masks[batch_indices, max_cost_idx[:,None], x_max_cur_pos[:,None]] = 1
        elif self.env_masks_mode == 1:
            #仅允许最大开销智能体返回仓库
            # 旅途中阶段：选出最大的旅行开销
            # 选出处于0-1阶段的智能体
            masked_cost = np.where(active_agents, self.costs, -np.inf)
            masked_cost_sl = masked_cost[batch_indices_1d]
            max_cost_idx = np.argmax(masked_cost_sl, axis=-1)
            # # 将最大开销的智能体的城市0的mask置为1，其他智能体的城市0mask为0
            # repeat_masks[batch_indices, :, 0][active_agents[batch_indices]]= 0
            # repeat_masks[batch_indices, max_cost_idx[:,None], 0] = 1
            # 仅允许最大开销的智能体留在原地
            x_max_cur_pos = self.cur_pos[batch_indices_1d, max_cost_idx]
            repeat_masks[batch_indices, max_cost_idx[:,None], x_max_cur_pos[:,None]] = 1
            # repeat_masks[batch_indices, max_cost_idx[:,None], cur_pos] = 1

            valid_mask = active_agents[batch_indices_1d]  # 形状 (K, A)
            valid_indices = self.cur_pos[batch_indices_1d]  # 形状 (K, A)

            # 使用高级索引直接赋值
            # x = repeat_masks[batch_indices, np.arange(A), valid_indices]
            repeat_masks[batch_indices, np.arange(A), valid_indices] = False
            # xx =  repeat_masks[batch_indices, np.arange(A), valid_indices]
            x_max_cur_pos = self.cur_pos[batch_indices_1d, max_cost_idx]
            repeat_masks[batch_indices, max_cost_idx[:,None], x_max_cur_pos[:,None]] = True

            #
            # min_cost_idx =  np.argmin(masked_cost_sl, axis=-1)
            # # 使用高级索引直接赋值
            # valid_mask = active_agents[batch_indices_1d]  # 形状 (K, A)
            # valid_indices = self.cur_pos[batch_indices_1d]  # 形状 (K, A)
            # repeat_masks[batch_indices, np.arange(A), valid_indices] = valid_mask
            #
            # x_min_cur_pos = self.cur_pos[batch_indices_1d, min_cost_idx]
            # repeat_masks[batch_indices, min_cost_idx[:,None], x_min_cur_pos[:,None]] = 0

        else:
            raise NotImplementedError
        #
        # # 未触发阶段：城市0的mask为0
        # repeat_masks[:,:,0][self.traj_stages == 0] = 0

        # 阶段>=2：全掩码关闭但保留depot
        repeat_masks[self.stage_2, 1:] = 0  # 对于stage_2为True的位置，将最后维度的1:之后位置置为0
        repeat_masks[self.stage_2, 0] = 1  # 对于stage_2为True的位置，将最后维度的0位置置为1

        self.salesmen_mask = repeat_masks
        # allB = np.all(~self.salesmen_mask, axis=-1)
        # idx = np.argwhere(allB)
        # assert len(idx)==0, "all actions is ban"
        a = np.all(~repeat_masks, axis=-1)
        if np.any(a):
            pass
        return self.salesmen_mask

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
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad_max_norm", type=float, default=1)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=5e-3)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--city_nums", type=int, default=50)
    parser.add_argument("--random_city_num", type=bool, default=True)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=0)
    parser.add_argument("--env_masks_mode", type=int, default=0,
                        help="0 for only the min cost  not allow back depot; 1 for only the max cost allow back depot")
    parser.add_argument("--eval_interval", type=int, default=500, help="eval  interval")
    parser.add_argument("--use_conflict_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_conflict_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_actions_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_city_encoder", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--use_agents_mask", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--use_city_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--agents_adv_rate", type=float, default=0.2, help="rate of adv between agents")
    parser.add_argument("--conflict_loss_rate", type=float, default=0.5 + 0.5, help="rate of adv between agents")
    parser.add_argument("--only_one_instance", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--save_model_interval", type=int, default=10000, help="save model interval")
    parser.add_argument("--seed", type=int, default=528, help="random seed")
    args = parser.parse_args()

    env_config = {
        "salesmen": args.agent_num,
        "cities": args.city_nums,
        "seed": None,
        "mode": 'rand',
        "env_masks_mode":args.env_masks_mode,
        "use_conflict_model": args.use_conflict_model
    }
    env = MTSPEnv_IDVR(
        env_config
    )

    from envs.GraphGenerator import GraphGenerator as GG

    # g = GG(1, env_config["cities"])
    # graph = g.generate(1, env_config["cities"], dim=2)

    from algorithm.RefAgent.AgentV2 import AgentV2 as Agent
    from model.RefModel.config import ModelConfig as Config

    agent = Agent(args, Config)
    agent.load_model(args.agent_id)
    # features_nb, actions_nb, actions_no_conflict_nb, returns_nb, individual_returns_nb, masks_nb, dones_nb = agent.run_batch(
    #     env, graph, env_config["salesmen"], 32)
    from utils.TensorTools import _convert_tensor
    import numpy as np
    import torch
    from envs.GraphGenerator import GraphGenerator as GG
    g =GG()
    batch_graph = g.generate(batch_size=args.batch_size,num=args.city_nums)
    states, info = env.reset(graph = batch_graph)
    salesmen_masks = info["salesmen_masks"]
    agent.reset_graph(batch_graph)
    done =False
    agent.run_batch_episode(env, batch_graph, args.agent_num, False, info={
                "use_conflict_model": args.use_conflict_model})