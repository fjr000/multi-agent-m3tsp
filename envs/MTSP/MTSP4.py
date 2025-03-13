import argparse
import copy
import math

import numpy as np
from typing import Tuple, List, Dict
import sys

sys.path.append("../")
from envs.MTSP.Config import Config
from envs.GraphGenerator import GraphGenerator as GG
from utils.GraphPlot import GraphPlot as GP
from model.NNN.RandomAgent import RandomAgent
import torch.nn.functional as F


class MTSPEnv:
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
        self.cities = 50
        self.salesmen = 5
        self.seed = None
        self.mode = "rand"
        self.problem_size = 1
        self.env_masks_mode = 0
        if config is not None:
            self.__parse_config(config)

        self.graph = None
        self.trajectories = None
        self.last_costs = None
        self.costs = None
        self.mask = None

        self.dim = 12
        self.step_count = 0
        self.step_limit = -1
        self.stay_still_limit = -1
        self.remain_stay_still_log = None

        self.traj_stages = None
        self.dones_step = None
        self.salesmen_masks = None
        self.actions = None
        self.ori_actions = None

        self.distance_scale = np.sqrt(2)

    def __parse_config(self, config: Dict):
        self.cities = config.get("cities", self.cities)
        self.salesmen = config.get("salesmen", self.salesmen)
        self.seed = config.get("seed", self.seed)
        self.mode = config.get("mode", self.mode)
        self.problem_size = config.get("N_aug", self.problem_size)
        self.env_masks_mode = config.get("env_masks_mode", self.env_masks_mode)

        if self.seed is not None:
            np.random.seed(self.seed)
        self.GG = GG(self.problem_size, self.cities, 2, self.seed)

    def __init(self, graph=None):
        """

        :param graph: -> self.graph [B, N ,2]
        :return:
        """
        if graph is not None:
            if len(graph.shape) == 2:
                self.graph = graph[np.newaxis,]
            elif len(graph.shape) == 3:
                self.graph = graph
            else:
                assert False
            self.problem_size = graph.shape[0]
        elif self.graph is None or self.mode == "rand":
            self.graph = self.GG.generate(self.problem_size, self.cities, 2)

        self.graph_matrix = self.GG.nodes_to_matrix(self.graph)
        # self.graph = self.graph - self.graph[0]

        self.trajectories = np.ones((self.problem_size,self.salesmen,self.cities+2), dtype=np.int32)

        # self.last_costs = np.zeros(self.salesmen)
        # self.costs = np.zeros(self.salesmen)
        self.costs = np.zeros((self.problem_size, self.salesmen), dtype=np.float64)
        self.mask = np.ones((self.problem_size, self.cities,), dtype=np.bool_)
        self.mask[:,0] = 0
        self.step_count = 0
        self.step_limit = self.cities
        self.remain_stay_still_log = np.zeros((self.problem_size, self.salesmen,), dtype=np.int32)
        self.traj_stages = np.zeros((self.problem_size, self.salesmen,), dtype=np.int32) # 0 -> prepare; 1 -> travelling; 2 -> finished; 3 -> stay depot
        self.dones_step = np.zeros((self.problem_size, self.salesmen), dtype=np.int32)
        self.dones = np.zeros(self.problem_size, dtype=np.bool_)
        self.states = np.empty((self.problem_size,self.salesmen, self.dim), dtype=np.float32)
        self.salesmen_masks = None
        self.actions = None
        self.ori_actions = None
        self.ori_actions_list = []
        self.actions_list = []
        self.salesmen_masks_list= []
        self.traj_stage_list= []
        self.distance_scale = self.cities / self.salesmen

    def _get_salesmen_states(self):

        B = self.problem_size
        N = self.cities
        A = self.salesmen

        pos = self.trajectories[:,:,self.step_count] # [B,A]
        depot_idx = 0
        cur_pos = pos-1

        # 生成批量索引 [B, A]
        batch_indices = np.arange(B)[:, None]  # [B, 1]
        dis_depot = self.graph_matrix[batch_indices,cur_pos,depot_idx]  # [B,A]
        cur_cost = self.costs  # [B,A]
        max_cost = np.max(self.costs, keepdims=True, axis=1).repeat(A,axis = 1)  # [B,A]
        diff_max_cost = (max_cost - cur_cost) / (max_cost + 1e-8) # [B,A]
        min_cost = np.min(self.costs, keepdims=True, axis=1).repeat(A,axis = 1)  # [B,A]
        diff_min_cost = (min_cost - cur_cost) / (max_cost + 1e-8) # [B,A]

        # # 直接选取当前节点对应的距离行 [B, A, N]
        selected_dists = self.graph_matrix[batch_indices, cur_pos, :]
        each_depot_dist = self.graph_matrix[:,0:1,:]
        selected_dists_depot = selected_dists + each_depot_dist
        masked_distances_depot = np.where(self.mask[:,None,:].repeat(A,axis = 1), selected_dists_depot, np.nan)
        average_distances_depot = np.nanmean(masked_distances_depot, axis = 2)
        max_distances_depot = np.nanmax(masked_distances_depot, axis=2)
        min_distances_depot = np.nanmin(masked_distances_depot, axis=2)

        remain_salesmen_num = np.count_nonzero(self.traj_stages < 2, keepdims=True, axis=1)
        remain_cities_num = np.count_nonzero(self.mask, keepdims=True,axis=1)
        # remain_salesmen_ratio = (remain_salesmen_num / self.salesmen).repeat(A,axis = -1)  # remain agents ratio
        remain_salesmen_city_ratio = remain_cities_num / np.maximum(remain_salesmen_num,1) / self.cities

        # rank = np.argsort(self.costs, axis=1) / self.salesmen
        sum_costs = np.sum(self.costs, axis=1, keepdims=True)  # 维度 [B,1]
        weighted_diff = sum_costs - self.salesmen * self.costs  # 广播计算 [B,A]
        # denominator = max(self.salesmen - 1, 1) * self.distance_scale
        denominator = max(self.salesmen - 1, 1)
        avg_diff_cost = weighted_diff / denominator

        self.states[..., 0] = depot_idx
        self.states[..., 1] = cur_pos

        self.states[..., 2] = cur_cost
        self.states[..., 3] = diff_max_cost
        self.states[..., 4] = diff_min_cost
        self.states[..., 5] = avg_diff_cost

        self.states[..., 6] = dis_depot
        self.states[..., 7] = average_distances_depot
        self.states[..., 8] = max_distances_depot
        self.states[..., 9] = min_distances_depot

        self.states[..., 10] = remain_salesmen_city_ratio.repeat(A,axis = -1)
        self.states[..., 11] = np.clip(self.traj_stages, a_min=0, a_max=2) - 1

        # self.states[..., 9] = 1 - rank

        return self.states

    def _get_distance(self, id1, id2):
        return np.sqrt(np.sum(np.square(self.graph[id1 - 1] - self.graph[id2 - 1])))

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
        batch_indices = np.arange(B)[np.sum(active_agents, axis=-1) > 1][:, np.newaxis]

        if self.env_masks_mode == 0:
            # 返回仓库优化 (新增阶段0排除)
            masked_cost = np.where(active_agents, self.costs, np.inf)
            masked_cost_sl = masked_cost[batch_indices.squeeze(1)]
            min_cost_idx =  np.argmin(masked_cost_sl, axis=-1)
            repeat_masks[batch_indices, :, 0] = 1
            repeat_masks[batch_indices, min_cost_idx[:,None], 0] = 0
            # # 仅不允许最大开销的智能体留在原地
            # 允许所有激活智能体留在原地
            cur_pos = self.trajectories[..., self.step_count]-1
            for b_idx, min_idx in zip(batch_indices.squeeze(1), min_cost_idx):
                active_agts = active_agents[b_idx]
                for agent_idx in range(active_agents.shape[1]):
                    if active_agts[agent_idx] and agent_idx != min_idx:
                        pos = cur_pos[b_idx, agent_idx]
                        repeat_masks[b_idx, agent_idx, pos] = 1
                    else:
                        pos = cur_pos[b_idx, agent_idx]
                        repeat_masks[b_idx, agent_idx, pos] = 0
            # cur_pos = self.trajectories[..., self.step_count]-1
            # x_cur = cur_pos[batch_indices.squeeze(1), max_cost_idx]
            # repeat_masks[batch_indices, max_cost_idx[:,None], x_max_cur_pos[:,None]] = 1
        elif self.env_masks_mode == 1:
            #仅允许最大开销智能体返回仓库
            # 旅途中阶段：选出最大的旅行开销
            # 选出处于0-1阶段的智能体
            masked_cost = np.where(active_agents, self.costs, -np.inf)
            masked_cost_sl = masked_cost[batch_indices.squeeze(1)]
            max_cost_idx = np.argmax(masked_cost_sl, axis=-1)
            # 将最大开销的智能体的城市0的mask置为1，其他智能体的城市0mask为0
            repeat_masks[batch_indices, :, 0]= 0
            repeat_masks[batch_indices, max_cost_idx[:,None], 0] = 1
            # 仅允许最大开销的智能体留在原地
            cur_pos = self.trajectories[..., self.step_count]-1
            x_max_cur_pos = cur_pos[batch_indices.squeeze(1), max_cost_idx]
            repeat_masks[batch_indices, max_cost_idx[:,None], x_max_cur_pos[:,None]] = 1
            # repeat_masks[batch_indices, max_cost_idx[:,None], cur_pos] = 1
        else:
            raise NotImplementedError

        # 未触发阶段：城市0的mask为0
        repeat_masks[self.traj_stages == 0, 0] = 0

        # 阶段>=2：全掩码关闭但保留depot
        stage_2 = self.traj_stages >= 2
        repeat_masks[stage_2, 1:] = 0
        repeat_masks[stage_2, 0] = 1

        self.salesmen_mask = repeat_masks
        return self.salesmen_mask

    def reset(self, config=None, graph=None):
        if config is not None:
            self.__parse_config(config)
        self.__init(graph)

        env_info = {
            "graph": self.graph,
            "salesmen": self.salesmen,
            "cities": self.cities,
            "mask": self.mask,
            "salesmen_masks": self._get_salesmen_masks()
        }

        return self._get_salesmen_states(), env_info

    def _get_reward(self):
        self.dones = np.all(self.traj_stages >=2, axis=1)
        self.done = np.all(self.dones, axis=0)
        # self.rewards = np.where(self.dones[:,None], -np.max(self.costs,keepdims=True, axis=1).repeat(self.salesmen, axis = 1), 0)
        self.rewards = np.where(self.dones, -np.max(self.costs, axis=1), 0)
        return self.rewards
    def deal_conflict(self, actions: np.ndarray):
        mp = {}
        new_actions = np.zeros_like(actions)

        for idx, act in enumerate(actions):
            if act != 1:
                if act in mp:
                    mp[act].append(idx)
                else:
                    mp.update({act: [idx]})
            else:
                new_actions[idx] = act

        for act, idxs in mp.items():
            min_idx = idxs[0]
            min_cost = self.costs[min_idx] + self._get_distance(self.trajectories[min_idx][-1], act)
            for idx in idxs[1:]:
                nxt_cost = self.costs[idx] + self._get_distance(self.trajectories[idx][-1], act)
                if nxt_cost < min_cost:
                    min_cost = nxt_cost
                    min_idx = idx
            new_actions[min_idx] = act

        return new_actions

    def deal_conflict_batch(self, actions: np.ndarray):
        """
        实现的动作冲突解决函数，跳过动作值为1的冲突

        参数:
        actions: np.ndarray, 形状为[B,A], B为批次数, A为智能体数量
        costs: np.ndarray, 形状为[B,A], 表示每个批次每个智能体的开销

        返回:
        resolved_actions: np.ndarray, 形状为[B,A], 解决冲突后的动作
        """
        B, A = actions.shape
        # 初始化结果数组为原始动作值
        resolved_actions = actions.copy()

        for b, acts in enumerate(resolved_actions):
            unique_act, unique_count = np.unique(acts, return_counts=True)
            for a, c in zip(unique_act, unique_count):
                if a <=1 or c == 1:
                    pass
                else:
                    idx = np.argwhere(acts == a)
                    min_cost = np.inf
                    min_id = -1
                    for i in idx:
                        if self.costs[b,i] < min_cost:
                            min_cost = self.costs[b,i]
                            min_id = i
                    for i in idx:
                        if i == min_id:
                            acts[i] = a
                        else:
                            acts[i] = self.trajectories[b,i,self.step_count]
        return resolved_actions



    def reset_actions(self, actions):
        # 获取当前最后位置 [B, A]
        cur_pos = self.trajectories[:, :, self.step_count]

        # 生成条件掩码 [B, A]
        same_pos_mask = (cur_pos == actions)

        # 批量更新停留计数 (向量化条件判断)
        self.remain_stay_still_log = np.where(
            same_pos_mask,
            self.remain_stay_still_log + 1,  # 条件为真时+1
            0  # 条件为假时重置为0
        )

        # 将保持静止的动作置本身 [B, A]
        actions = np.where(same_pos_mask, cur_pos, actions)
        actions = np.where(actions == 0, self.trajectories[...,self.step_count], actions)

        return actions

    def step(self, ori_actions: np.ndarray):

        try:
            actions = self.reset_actions(ori_actions)
            actions = self.deal_conflict_batch(actions)
            self.actions = actions
            self.step_count += 1
            self.trajectories[..., self.step_count] = actions
        except Exception as e:
            np.set_printoptions(threshold=np.inf)
            print(f"trajectories:{self.trajectories} ")
            print(f"mask:{self.mask} ")
            print(f"traj_stages:{self.traj_stages} ")
            with open("error log", 'w') as f:
                print(f"trajectories:{self.trajectories} ", file=f)
                print(f"mask:{self.mask} ", file=f)
                print(f"traj_stages:{self.traj_stages} ",file=f)
                print(f"actions:{actions} ", file=f)
                idx = np.argwhere(~ self.dones)
                print(f"idx:{idx}")
                print(f"tra:{self.trajectories[idx]}", file=f)
                print(f"mask:{self.mask[idx]}", file=f)
                print(f"traj_stages:{self.traj_stages[idx]}", file=f)
                print(f"actions:{self.actions[idx]}", file=f)
        # 生成行索引 [B*A]
        batch_indices = np.repeat(np.arange(self.problem_size), self.salesmen)

        # 展平列索引 [B*A]
        col_indices = actions.ravel()-1
        # 批量置零操作
        self.mask[batch_indices, col_indices] = 0

        last_pos = self.trajectories[..., self.step_count-1]-1
        cur_pos = self.trajectories[...,self.step_count]-1

        self.costs += self.graph_matrix[
            np.arange(self.problem_size)[:,None],
            last_pos,
            cur_pos,
        ]
        self._get_reward()

        batch_complete = np.all(~self.mask, axis=1)
        # 判断哪些批次所有城市已访问完成 [B]

        self.traj_stages = np.where(
            ((last_pos != cur_pos)
            &
            ((last_pos == 0)|(cur_pos == 0)))
            |
            (batch_complete[:, None]),
            # |
            # (np.all(actions == 1)),
            self.traj_stages + 1,  # 满足条件时阶段+1
            self.traj_stages  # 否则保持原值
        )

        # 找出需要处理的旅行商 [B, A]
        need_process = (
                batch_complete[:, None] &  # 批次完成标记
                (self.traj_stages == 0)    # 未出发状态
        )

        # 批量更新轨迹和状态
        if np.any(need_process):
            # 更新状态
            self.traj_stages[need_process] = 2

        self.mask[batch_complete,0] =1

        info = {
            "mask": self.mask,
            "salesmen_masks": self._get_salesmen_masks(),
        }

        # self.ori_actions_list.append(ori_actions)
        # self.actions_list.append(actions)
        self.salesmen_masks_list.append(info["salesmen_masks"])
        # self.traj_stage_list.append(self.traj_stages)

        if self.done:

            # valid = self.check_array_structure()
            # ca = self.compress_array()
            # t = self.convert_to_list(ca)

            info.update(
                {
                    "trajectories": self.trajectories[...,:self.step_count+1],
                    "costs": self.costs,
                }
            )

        return self._get_salesmen_states(), self.rewards, self.done, info

    def draw(self, graph, cost, trajectory, used_time=0, agent_name="agent", draw=True):
        from utils.GraphPlot import GraphPlot as GP
        graph_plot = GP()
        if agent_name == "or_tools":
            one_first = False
        else:
            one_first = True
        return graph_plot.draw_route(graph, trajectory, draw=draw,
                                     title=f"{agent_name}_cost:{cost:.5f}_time:{used_time:.3f}", one_first=one_first)

    def draw_multi(self, graph, costs, trajectorys, used_times=(0,), agent_names=("agents",), draw=True):
        figs = []
        for c, t, u, a in zip(costs, trajectorys, used_times, agent_names):
            figs.append(self.draw(graph, c, t, u, a, False))
        from utils.GraphPlot import GraphPlot as GP
        graph_plot = GP()
        fig = graph_plot.combine_figs(figs)
        if draw:
            fig.show()
        return fig

    @staticmethod
    def compress_adjacent_duplicates_optimized(arr):
        """
        优化版本：合并 B 和 A 维度，减少循环层数
        输入形状：[B, A, T]
        输出形状：[B*[A*[]]]
        """
        B, A, T = arr.shape
        if T == 0:
            return [[[] for _ in range(A)] for _ in range(B)]

        # 合并 B 和 A 维度，转化为二维数组 [B*A, T]
        arr_2d = arr.reshape(-1, T)

        # 向量化生成掩码（相邻元素不同时标记为 True）
        mask = np.ones_like(arr_2d, dtype=bool)
        if T > 1:
            mask[:, 1:] = (arr_2d[:, 1:] != arr_2d[:, :-1])

        # 提取非重复元素并转换为列表（单层循环）
        compressed_2d = [arr_2d[i, mask[i]].tolist() for i in range(arr_2d.shape[0])]

        # 重新分割为 [B, A] 结构
        return [compressed_2d[i * A: (i + 1) * A] for i in range(B)]

    def check_array_structure(self) -> np.ndarray:
        B, A, T = self.trajectories.shape
        ones_mask = (self.trajectories == 1)

        # 条件1：检查1是否仅出现在开头和结尾
        first_one = np.argmax(ones_mask, axis=-1)
        last_one = T - 1 - np.argmax(ones_mask[..., ::-1], axis=-1)
        valid_ones = (first_one < last_one) & np.all(
            ones_mask & (np.arange(T) < first_one[..., None]) | (np.arange(T) > last_one[..., None]), axis=-1)

        # 条件2：检查相同的数值是否仅相邻出现
        diff_mask = np.diff(self.trajectories, axis=-1) != 0
        valid_unique = np.all(diff_mask | (self.trajectories == 1)[..., 1:], axis=-1)

        return np.all(valid_ones & valid_unique, axis=-1)

    def convert_to_list(self, arr: np.ndarray) -> list:
        B, A, T = arr.shape
        return [[list(filter(lambda x: x != 0, arr[b, a])) for a in range(A)] for b in range(B)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--agent_num", type=int, default=3)
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
    parser.add_argument("--batch_size", type=float, default=2)
    parser.add_argument("--city_nums", type=int, default=5)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=0)
    parser.add_argument("--env_masks_mode", type=int, default=0, help="0 for only the min cost  not allow back depot; 1 for only the max cost allow back depot")
    parser.add_argument("--eval_interval", type=int, default=100, help="eval  interval")
    args = parser.parse_args()

    env_config = {
        "salesmen": args.agent_num,
        "cities": args.city_nums,
        "seed": None,
        "mode": 'rand',
        "env_masks_mode":args.env_masks_mode
    }
    env = MTSPEnv(
        env_config
    )

    from envs.GraphGenerator import GraphGenerator as GG

    # g = GG(1, env_config["cities"])
    # graph = g.generate(1, env_config["cities"], dim=2)

    from algorithm.DNN5.AgentV1 import AgentV1 as Agent
    from model.n4Model.config import Config

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
    while not done:
        states_t = _convert_tensor(states, device=agent.device)
        salesmen_masks_t = _convert_tensor(salesmen_masks, device=agent.device)
        acts, acts_no_conflict, act_logp, agents_logp, act_entropy, agt_entropy = agent.predict(states_t, salesmen_masks_t)
        states, r, done, info = env.step(acts_no_conflict+1)
        salesmen_masks = info["salesmen_masks"]

