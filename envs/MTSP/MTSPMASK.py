import argparse
import copy
import math
import time

import numpy as np
from typing import Tuple, List, Dict
import sys
from envs.MTSP.MTSPBase import MTSPEnv as Env

sys.path.append("../")
from envs.MTSP.Config import Config
from envs.GraphGenerator import GraphGenerator as GG
from utils.GraphPlot import GraphPlot as GP
from model.NNN.RandomAgent import RandomAgent
import torch.nn.functional as F


class MTSPEnv(Env):
    """
    1. 智能体离开仓库后，再次抵达仓库后不允许其离开仓库，salesman mask中只有depot
    2. 允许智能体中当前旅行成本最高M-1个返回仓库，由最后一个智能体保证完备性（还是通过允许最远的返回仓库？）
    3. 允许智能体静止在原地（限制与否？）
    4. 支持batch的环境， batch中单个示例结束后可送入env得到原有的state

    config: dict
        - salesmen  int
        - cities    int
            - salesmen <= cities;
        - mode ['fixed'|'rand'] str
        - seed      int
    """

    def __init__(self, config: Dict = None):
        super(MTSPEnv, self).__init__(config)

    def _get_salesmen_masks(self):
        A = self.salesmen

        # 初始化批量掩码 [B, A, N]
        repeat_masks = np.tile(self.mask[:, None, :], (1, A, 1))

        if self.env_masks_mode == 0:
            active_agents = self.traj_stages == 1
            batch_indices_1d = self.batch_ar[np.sum(active_agents, axis=-1) > 1]
            batch_indices = batch_indices_1d[:, None]

            # 返回仓库优化 (新增阶段0排除)
            masked_cost = np.where(active_agents, self.costs, np.inf)
            masked_cost_sl = masked_cost[batch_indices_1d]
            min_cost_idx = np.argmin(masked_cost_sl, axis=-1)
            # 仅不允许最小开销的智能体留在原地
            # 允许所有激活智能体留在原地
            # 筛选有效批次的 mask 和 indices
            valid_mask = active_agents[batch_indices_1d]  # 形状 (K, A)
            valid_indices = self.cur_pos[batch_indices_1d]  # 形状 (K, A)

            # 使用高级索引直接赋值
            repeat_masks[batch_indices, np.arange(A), valid_indices] = valid_mask
            x_min_cur_pos = self.cur_pos[batch_indices_1d, min_cost_idx]
            repeat_masks[batch_indices, min_cost_idx[:, None], x_min_cur_pos[:, None]] = 0

        elif self.env_masks_mode == 1:
            active_agents = self.traj_stages == 1
            batch_indices_1d = self.batch_ar[np.sum(active_agents, axis=-1) > 1]
            batch_indices = batch_indices_1d[:, None]
            #仅允许最大开销智能体返回仓库
            # 旅途中阶段：选出最大的旅行开销
            # 选出处于0-1阶段的智能体
            masked_cost = np.where(active_agents, self.costs, -np.inf)
            masked_cost_sl = masked_cost[batch_indices_1d]
            max_cost_idx = np.argmax(masked_cost_sl, axis=-1)
            # 仅允许最大开销的智能体留在原地
            x_max_cur_pos = self.cur_pos[batch_indices_1d, max_cost_idx]
            repeat_masks[batch_indices, max_cost_idx[:, None], x_max_cur_pos[:, None]] = 1

            valid_indices = self.cur_pos[batch_indices_1d]  # 形状 (K, A)

            # 使用高级索引直接赋值
            repeat_masks[batch_indices, np.arange(A), valid_indices] = False
            x_max_cur_pos = self.cur_pos[batch_indices_1d, max_cost_idx]
            repeat_masks[batch_indices, max_cost_idx[:, None], x_max_cur_pos[:, None]] = True

        elif self.env_masks_mode == 2:
            active_agents = self.traj_stages == 1
            batch_indices_1d = self.batch_ar[np.sum(active_agents, axis=-1) > 1]
            batch_indices = batch_indices_1d[:, None]
            # 结合回仓库距离
            dis_depot = self.graph_matrix[batch_indices, self.cur_pos[batch_indices_1d], 0]  # [B,A]
            expect_dis = self.costs[batch_indices_1d] + dis_depot

            # 返回仓库优化 (新增阶段0排除)
            masked_cost = np.where(active_agents[batch_indices_1d], expect_dis, np.inf)
            min_cost_idx = np.argmin(masked_cost, axis=-1)
            # 仅不允许最小开销的智能体留在原地
            # 允许所有激活智能体留在原地
            # 筛选有效批次的 mask 和 indices
            valid_mask = active_agents[batch_indices_1d]  # 形状 (K, A)
            valid_indices = self.cur_pos[batch_indices_1d]  # 形状 (K, A)

            # 使用高级索引直接赋值
            repeat_masks[batch_indices, np.arange(A)[None, :], valid_indices] = valid_mask
            x_min_cur_pos = self.cur_pos[batch_indices_1d, min_cost_idx]
            repeat_masks[batch_indices_1d, min_cost_idx, x_min_cur_pos] = 0

        elif self.env_masks_mode == 3:
            active_agents = self.traj_stages == 1
            batch_indices_1d = self.batch_ar[np.sum(active_agents, axis=-1) > 1]
            batch_indices = batch_indices_1d[:, None]

            dis_depot = self.graph_matrix[batch_indices, self.cur_pos[batch_indices_1d], 0]  # [B,A]
            expect_dis = self.costs[batch_indices_1d] + dis_depot
            # 返回仓库优化 (新增阶段0排除)
            masked_cost = np.where(active_agents[batch_indices_1d], expect_dis, np.inf)
            max_cost_idx = np.argmax(masked_cost, axis=-1)
            # 仅允许最大开销的智能体留在原地
            x_max_cur_pos = self.cur_pos[batch_indices_1d, max_cost_idx]
            repeat_masks[batch_indices_1d, max_cost_idx, x_max_cur_pos] = True

        elif self.env_masks_mode == 4:

            active_agents = self.traj_stages <= 1
            batch_indices_1d = self.batch_ar[np.sum(active_agents, axis=-1) > 1]
            batch_indices = batch_indices_1d[:, None]

            dis_depot = self.graph_matrix[batch_indices, self.cur_pos[batch_indices_1d], 0]  # [B,A]
            expect_dis = self.costs[batch_indices_1d] + dis_depot  # [B,A]

            # 对每个批次掩码活跃智能体的开销
            masked_costs = np.where(active_agents[batch_indices_1d], expect_dis, np.nan)

            # 批量计算均值和标准差 (忽略nan值)
            mean_costs = np.nanmean(masked_costs, axis=1, keepdims=True)  # [B,1]
            # std_costs = np.nanstd(masked_costs, axis=1, keepdims=True)  # [B,1]

            # 设置动态阈值
            # alpha = 0.5 * (np.count_nonzero(self.mask[batch_indices_1d], axis=-1) / self.cities)
            thresholds = mean_costs  # [B,1]
            # thresholds = mean_costs + alpha[:,None] * std_costs  # [B,1]
            # 计算stay_masks: 哪些智能体超过阈值可以留在原地
            if len(mean_costs) > 0:
                stay_masks = masked_costs > thresholds  # [B,A] 布尔数组
                # 找出所有满足条件的(batch_idx, agent_idx)对
                # 返回一个(N,2)数组，其中N是True值的数量，每行包含[批次索引,智能体索引]
                stay_indices = np.argwhere(stay_masks)

                if len(stay_indices) > 0:  # 确保有满足条件的智能体
                    # 提取批次和智能体索引
                    b_indices_rel = stay_indices[:, 0]  # 相对于batch_indices_1d的批次索引
                    a_indices = stay_indices[:, 1]  # 智能体索引

                    # 获取实际批次索引
                    b_indices_abs = batch_indices_1d[b_indices_rel]  # 绝对批次索引

                    # 获取这些智能体的当前位置
                    positions = self.cur_pos[b_indices_abs, a_indices]

                    # 一次性更新repeat_masks
                    repeat_masks[b_indices_abs, a_indices, positions] = True

            min_cost_idx = np.argmin(masked_costs, axis=-1)
            # 使用高级索引直接赋值
            x_min_cur_pos = self.cur_pos[batch_indices_1d, min_cost_idx]
            repeat_masks[batch_indices_1d, min_cost_idx, x_min_cur_pos] = False
        elif self.env_masks_mode == 5:
            active_agents = self.traj_stages <= 1  # [B,A]
            batch_indices_1d = self.batch_ar[np.sum(active_agents, axis=-1) > 1]  # [B,]
            batch_indices = batch_indices_1d[:, None]  # [B,A]

            cur_costs = self.costs[batch_indices_1d]  # [B,A]
            cur_pos = self.cur_pos[batch_indices_1d]  # [B,A]
            dis_depot = self.graph_matrix[batch_indices, cur_pos, 0]  # [B,A]
            expect_dis = cur_costs + dis_depot  # [B,A]
            max_expect_dis = np.max(expect_dis, axis=-1, keepdims=True)

            selected_dists = self.graph_matrix[batch_indices, cur_pos]  # [B,A,N]
            each_depot_dist = self.graph_matrix[batch_indices_1d, 0:1, :]  # Depot to all cities [B,1,N]
            selected_dists_depot = cur_costs[..., None] + selected_dists + each_depot_dist  # [B,A,N]
            masked_dist_depot = np.where(self.mask[batch_indices_1d, None, :], selected_dists_depot, np.inf)
            min_dist_depot = np.min(masked_dist_depot, axis=2)  # [B,A]
            min_min_dist_depot = np.min(min_dist_depot, axis=1, keepdims=True)
            allow_stay = ((min_dist_depot >= max_expect_dis) & (min_dist_depot != min_min_dist_depot))

            repeat_masks[batch_indices, np.arange(A)[None,], cur_pos] = allow_stay
        elif self.env_masks_mode == 6:
            active_agents = self.traj_stages <= 1  # [B,A]
            batch_indices_1d = self.batch_ar[np.sum(active_agents, axis=-1) > 1]  # [B,]
            batch_indices = batch_indices_1d[:, None]  # [B,A]

            cur_costs = self.costs[batch_indices_1d]  # [B,A]
            cur_pos = self.cur_pos[batch_indices_1d]  # [B,A]
            dis_depot = self.graph_matrix[batch_indices, cur_pos, 0]  # [B,A]
            expect_dis = cur_costs + dis_depot  # [B,A]
            max_expect_dis = np.max(expect_dis, axis=-1, keepdims=True)

            selected_dists = self.graph_matrix[batch_indices, cur_pos]  # [B,A,N]
            each_depot_dist = self.graph_matrix[batch_indices_1d, 0:1, :]  # Depot to all cities [B,1,N]
            selected_dists_depot = cur_costs[..., None] + selected_dists + each_depot_dist  # [B,A,N]
            masked_dist_depot = np.where(self.mask[batch_indices_1d, None, :], selected_dists_depot, np.inf)
            min_dist_depot = np.min(masked_dist_depot, axis=2)  # [B,A]
            min_min_dist_depot = np.min(min_dist_depot, axis=1, keepdims=True)
            # allow_stay = ((min_dist_depot >= max_expect_dis) & (min_dist_depot != min_min_dist_depot))
            allow_stay = (min_dist_depot != min_min_dist_depot)

            repeat_masks[batch_indices, np.arange(A)[None,], cur_pos] = allow_stay
        else:
            raise NotImplementedError
        #
        # # 未触发阶段：城市0的mask为0
        # repeat_masks[:, :, 0][self.traj_stages == 0] = 0

        # 阶段>=2：全掩码关闭但保留depot
        repeat_masks[self.stage_2, 1:] = 0  # 对于stage_2为True的位置，将最后维度的1:之后位置置为0
        repeat_masks[self.stage_2, 0] = 1  # 对于stage_2为True的位置，将最后维度的0位置置为1

        self.salesmen_mask = repeat_masks

        return self.salesmen_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--agent_num", type=int, default=5)
    parser.add_argument("--agent_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=9e-5)
    parser.add_argument("--grad_max_norm", type=float, default=1.0)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=5e-3)
    parser.add_argument("--batch_size", type=float, default=128)
    parser.add_argument("--city_nums", type=int, default=20)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=0)
    parser.add_argument("--env_masks_mode", type=int, default=5,
                        help="0 for only the min cost  not allow back depot; 1 for only the max cost allow back depot")
    parser.add_argument("--eval_interval", type=int, default=100, help="eval  interval")
    parser.add_argument("--use_conflict_model", type=bool, default=True, help="0:not use;1:use")
    args = parser.parse_args()

    env_config = {
        "salesmen": args.agent_num,
        "cities": args.city_nums,
        "seed": None,
        "mode": 'rand',
        "env_masks_mode": args.env_masks_mode,
        "use_conflict_model": args.use_conflict_model
    }
    env = MTSPEnv(
        env_config
    )

    from envs.GraphGenerator import GraphGenerator as GG

    # g = GG(1, env_config["cities"])
    # graph = g.generate(1, env_config["cities"], dim=2)

    from algorithm.DNN5.AgentV5 import Agent as Agent
    from model.n4Model.config import Config

    agent = Agent(args, Config)
    agent.load_model(args.agent_id)
    # features_nb, actions_nb, actions_no_conflict_nb, returns_nb, individual_returns_nb, masks_nb, dones_nb = agent.run_batch(
    #     env, graph, env_config["salesmen"], 32)
    from utils.TensorTools import _convert_tensor
    import numpy as np
    import torch
    from envs.GraphGenerator import GraphGenerator as GG

    g = GG()
    batch_graph = g.generate(batch_size=args.batch_size, num=args.city_nums)

    states, info = env.reset(graph=batch_graph)
    salesmen_masks = info["salesmen_masks"]
    st = time.time_ns()
    agent.reset_graph(batch_graph, args.agent_num)
    done = False
    for i in range(100):
        agent.run_batch_episode(env, batch_graph, args.agent_num, False, info={
            "use_conflict_model": args.use_conflict_model})
    ed = time.time_ns()
    print((ed - st) / 1e9)
