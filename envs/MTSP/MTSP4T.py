# ================== MTSPEnvTorch.py ==================
"""
Mixed-precision 版 MTSP 环境
接口与旧版保持一致，内部主要张量均使用 float16 以降低显存
"""
from __future__ import annotations
import argparse, math
from typing import Dict, Tuple, List

import numpy as np  # 仅在出/入口使用
import torch, torch.nn.functional as F
import random

# -----------------------------------------------------
# 公共工具
# -----------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
FP_DTYPE = torch.float32  # 存储用
ACC_DTYPE = torch.float32  # 计算用(距 离/代价)


def _np2th(x, dtype: torch.dtype | None = None):
    """
    numpy / list / scalar → torch.tensor（放到默认 device）
    对浮点统一转 FP_DTYPE；整型 / bool 不变
    """
    tgt_dtype = dtype or FP_DTYPE
    if isinstance(x, torch.Tensor):
        return x.to(device)

    if isinstance(x, np.ndarray):
        if x.dtype == np.bool_:
            return torch.as_tensor(x, device=device, dtype=torch.bool)
        if np.issubdtype(x.dtype, np.integer):
            return torch.as_tensor(x, device=device, dtype=torch.long)
        # 浮点
        return torch.as_tensor(x, device=device, dtype=FP_DTYPE)

    # python list / scalar
    return torch.as_tensor(x, device=device, dtype=tgt_dtype)


def _th2np(x):
    "torch → numpy（搬回 cpu，detach）"
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


# -----------------------------------------------------
# 原始依赖（保持不变）
# -----------------------------------------------------
from envs.GraphGenerator import GraphGenerator as GG
from utils.GraphPlot import GraphPlot as GP


# =====================================================
#  Mixed-precision MTSP 环境
# =====================================================
class MTSPEnv:
    """
    与原 numpy 版说明一致，此处省略注释
    """

    # -------------------------------------------------
    # 1. 初始化 / 配置
    # -------------------------------------------------
    def __init__(self, config: Dict | None = None):
        # === 与原版保持一致的成员变量（能少动就少动） ===
        self.stage_2: torch.Tensor | None = None
        self.cur_pos: torch.Tensor | None = None
        self.cities = 50
        self.salesmen = 5
        self.seed = None
        self.mode = "rand"
        self.problem_size = 1
        self.env_masks_mode = 0
        self.use_conflict_model = False

        # 维度常数
        self.dim = 13

        # 其余运行时变量
        self.step_count = 0
        self.graph: torch.Tensor | None = None
        self.graph_matrix: torch.Tensor | None = None
        self.trajectories: np.ndarray | None = None
        self.costs: torch.Tensor | None = None
        self.mask: torch.Tensor | None = None
        self.traj_stages: torch.Tensor | None = None
        self.salesmen_mask: torch.Tensor | None = None
        self.actions: torch.Tensor | None = None
        self.ori_actions: torch.Tensor | None = None
        self.conflict_count_exp: torch.Tensor | None = None
        self.conflict_count: torch.Tensor | None = None
        self.dones: torch.Tensor | None = None
        self.states: torch.Tensor | None = None
        self.batch_ar: torch.Tensor | None = None
        self.batch_salesmen_ar: torch.Tensor | None = None
        self.rewards: torch.Tensor | None = None
        self.done: bool = False
        self._min_distance: np.ndarray | None = None

        self.device = device

        if config is not None:
            self._parse_config(config)

    # -------------------------------------------------
    # 2. 解析外部配置
    # -------------------------------------------------
    def _parse_config(self, config: Dict):
        self.cities = config.get("cities", self.cities)
        self.salesmen = config.get("salesmen", self.salesmen)
        self.seed = config.get("seed", self.seed)
        self.mode = config.get("mode", self.mode)
        self.problem_size = config.get("N_aug", self.problem_size)
        self.env_masks_mode = config.get("env_masks_mode", self.env_masks_mode)
        self.use_conflict_model = config.get("use_conflict_model", self.use_conflict_model)

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        self.GG = GG(self.problem_size, self.cities, 2, self.seed)

    # -------------------------------------------------
    # 3. 初始化环境状态
    # -------------------------------------------------
    def _init(self, graph: np.ndarray | torch.Tensor | None = None):
        """
        建图 / 初始化各种张量
        """
        # -------- 3.1 读取 / 生成图 ----------
        if graph is not None:
            tg = _np2th(graph)  # 已转半精度
            if tg.ndim == 2:
                tg = tg.unsqueeze(0)
            elif tg.ndim != 3:
                raise ValueError("graph must be [N,2] or [B,N,2]")
            self.graph = tg  # [B,N,2]
            self.problem_size = self.graph.size(0)

        elif self.graph is None or self.mode == "rand":
            g_np = self.GG.generate(self.problem_size, self.cities, 2)
            self.graph = _np2th(g_np)  # [B,N,2], 半精度

        # 距离矩阵 [B,N,N] —— 生成后直接 cast
        # graph_cal = self.graph.float()
        graph_cal = self.graph.to(dtype=torch.float64)
        self.graph_matrix = torch.cdist(graph_cal, graph_cal).to(dtype=torch.float32)

        # -------- 3.2 各种运行时张量 ----------
        B, A, N = self.problem_size, self.salesmen, self.cities
        self.step_count = 0

        self.trajectories = np.ones((B, A, N + 1), dtype=np.int_)
        self.cur_pos = torch.zeros(B, A, dtype=torch.long, device=self.device)
        self.last_pos = self.cur_pos.clone().detach()

        self.costs = torch.zeros(B, A, dtype=ACC_DTYPE, device=self.device)
        self.mask = torch.ones(B, N, dtype=torch.bool, device=self.device)
        self.mask[:, 0] = False  # depot 不可选

        self.traj_stages = torch.zeros(B, A, dtype=torch.long, device=self.device)
        self.stage_2 = torch.zeros_like(self.traj_stages, dtype=torch.bool)

        self.batch_ar = torch.arange(B, device=self.device)
        self.batch_salesmen_ar = self.batch_ar.repeat_interleave(A)  # [B*A]

        self.conflict_count_exp = torch.zeros(B, A, dtype=torch.long, device=self.device)
        self.conflict_count = torch.zeros_like(self.conflict_count_exp)

        # min distance (排除 0)
        gm_tmp = torch.where(
            torch.isclose(self.graph_matrix, torch.tensor(0., device=self.device, dtype=FP_DTYPE)),
            torch.tensor(float('inf'), device=self.device, dtype=FP_DTYPE),
            self.graph_matrix
        )
        self._min_distance = torch.min(gm_tmp.view(B, -1), dim=-1).values.cpu().numpy()  # [B]
        del gm_tmp
    # -------------------------------------------------
    # 4. 计算智能体 state
    # -------------------------------------------------
    @torch.no_grad()
    def _get_salesmen_states(self) -> np.ndarray:
        B, A, N = self.problem_size, self.salesmen, self.cities
        depot_idx = 0

        batch_indices = self.batch_ar[:, None]
        city_mask = self.mask[:,None,:]
        cur_pos_dists = self.graph_matrix[batch_indices, self.cur_pos, :]
        remain_cities = torch.sum(self.mask, dim=1, keepdim=True)  # [B,1]

        # === 2. 智能体级特征 ===
        # 成本特征
        max_cost = torch.max(self.costs, dim=1, keepdim=True)[0]  # [B,1]
        min_cost = torch.min(self.costs, dim=1, keepdim=True)[0]  # [B,1]
        depot_dist = self.graph_matrix[batch_indices, self.cur_pos, 0]
        expect_dist = self.costs + depot_dist
        # max_expect_dist = np.max(expect_dist, axis=1, keepdims=True)  # 若现在结束的开销

        # 最近-最远点距离
        masked_dists_inf = torch.where(city_mask, cur_pos_dists, torch.inf)
        masked_dists_ninf = torch.where(city_mask, cur_pos_dists, -torch.inf)
        max_dists = torch.max(masked_dists_ninf, dim=2)[0]
        min_dists = torch.min(masked_dists_inf, dim=2)[0]

        # 远视距离 cur -》 next -》 depot
        selected_dists = cur_pos_dists  # [B,A,N]
        each_depot_dist = self.graph_matrix[:, 0:1, :]  # Depot to all cities [B,1,N]
        selected_dists_depot = selected_dists + each_depot_dist  # [B,A,N]


        # 最近-最远远视距离
        masked_dist_depot_inf = torch.where(city_mask, selected_dists_depot, torch.inf)
        masked_dist_depot_ninf = torch.where(city_mask, selected_dists_depot, -torch.inf)

        max_dist_depot = torch.max(masked_dist_depot_ninf, dim=2)[0]  # [B,A]
        min_dist_depot = torch.min(masked_dist_depot_inf, dim=2)[0]  # [B,A]

        max_max_dist_depot = torch.max(self.costs + max_dist_depot, dim=1, keepdim=True)[0]

        # === 4. 全局任务特征 ===
        progress = 1 - remain_cities / (N - 1)  # [B,1]
        scale = max_max_dist_depot
        allow_stayup = self.salesmen_mask[batch_indices, torch.arange(A)[None,], self.cur_pos]

        states32 = torch.empty(B, A, self.dim, dtype=ACC_DTYPE, device=self.device)
        states32[..., 0] = depot_idx
        states32[..., 1] = self.cur_pos
        states32[..., 2] = self.costs / scale
        states32[..., 3] = depot_dist / scale
        states32[..., 4] = max_dists / scale
        states32[..., 5] = min_dists / scale
        states32[..., 6] = max_dist_depot / scale
        states32[..., 7] = min_dist_depot / scale
        states32[..., 8] = allow_stayup.float()
        states32[..., 9] = max_cost / scale
        states32[..., 10] = min_cost / scale
        states32[..., 11] = max_max_dist_depot
        states32[..., 12] = progress
        return states32.float().cpu().numpy()  # 返 float32

    # -------------------------------------------------
    # 5. 距离查询（冲突处理里会用）
    # -------------------------------------------------
    def _get_distance(self, id1: int, id2: int) -> float:
        # 半精度足够，但为了避免 python sqrt 的类型问题转为 float32
        p1 = self.graph[0, id1 - 1].float()
        p2 = self.graph[0, id2 - 1].float()
        return math.sqrt(float(torch.sum((p1 - p2) ** 2)))

    # -------------------------------------------------
    # 6. 生成 salesmen mask（与原版完全等价）
    # -------------------------------------------------
    @torch.no_grad()
    def _get_salesmen_masks(self) -> np.ndarray:
        B, N = self.mask.shape
        A = self.salesmen

        repeat_masks = self.mask[:, None, :].repeat(1, A, 1)  # [B,A,N]
        active_agents = self.traj_stages == 1  # [B,A]
        batch_idx1d = torch.nonzero(torch.sum(active_agents, dim=1) > 1).squeeze(1)
        batch_idx = batch_idx1d[:, None]  # [K,1]

        if self.env_masks_mode in (0, 2):
            dis_depot = self.graph_matrix[batch_idx, self.cur_pos[batch_idx1d], 0] \
                if self.env_masks_mode >= 2 else 0.
            expect_dis = self.costs[batch_idx1d] + dis_depot
            masked_cost = torch.where(active_agents[batch_idx1d], expect_dis,
                                      torch.tensor(float('inf'), device=self.device, dtype=FP_DTYPE))
            min_cost_idx = torch.argmin(masked_cost, dim=-1)
            repeat_masks[batch_idx, :, 0] = True
            repeat_masks[batch_idx, min_cost_idx[:, None], 0] = False
            # allow stay
            repeat_masks[batch_idx, torch.arange(A, device=self.device)[None, :],
            self.cur_pos[batch_idx1d]] = active_agents[batch_idx1d]
            repeat_masks[batch_idx, min_cost_idx[:, None],
            self.cur_pos[batch_idx1d, min_cost_idx]] = False

        elif self.env_masks_mode in (1, 3):
            dis_depot = self.graph_matrix[batch_idx, self.cur_pos[batch_idx1d], 0] \
                if self.env_masks_mode >= 3 else 0.
            expect_dis = self.costs[batch_idx1d] + dis_depot
            masked_cost = torch.where(active_agents[batch_idx1d], expect_dis,
                                      torch.tensor(float('-inf'), device=self.device, dtype=FP_DTYPE))
            max_cost_idx = torch.argmax(masked_cost, dim=-1)
            repeat_masks[batch_idx, :, 0].masked_fill_(active_agents[batch_idx], False)
            repeat_masks[batch_idx, max_cost_idx[:, None], 0] = True
            repeat_masks[batch_idx, max_cost_idx[:, None],
            self.cur_pos[batch_idx1d, max_cost_idx]] = True
        else:
            raise NotImplementedError

        # stage ≥ 2: 仅允许 depot
        if self.stage_2.any():
            repeat_masks[self.stage_2] = False
            b_idx, a_idx = torch.nonzero(self.stage_2, as_tuple=True)
            repeat_masks[b_idx, a_idx, 0] = True

        self.salesmen_mask = repeat_masks
        return _th2np(repeat_masks)

    # -------------------------------------------------
    # 7. reset
    # -------------------------------------------------
    @torch.no_grad()
    def reset(self, config: Dict | None = None, graph=None):
        if config is not None:
            self._parse_config(config)
        self._init(graph)

        env_info = {
            "graph": _th2np(self.graph.float()),  # float32 给外界
            "salesmen": self.salesmen,
            "cities": self.cities,
            "mask": _th2np(self.mask),
            "salesmen_masks": self._get_salesmen_masks(),
            "masks_in_salesmen": None,
            "min_distance": self._min_distance,
            "dones": None
        }
        return self._get_salesmen_states(), env_info

    # -------------------------------------------------
    # 8. reward / done 判断
    # -------------------------------------------------
    def _get_reward(self):
        self.dones = torch.all(self.stage_2, dim=1)  # [B]
        self.done = bool(torch.all(self.dones))
        max_cost = torch.max(self.costs, dim=1).values  # [B]
        self.rewards = torch.where(self.dones, -max_cost, torch.zeros_like(max_cost))
        return self.rewards

    # -------------------------------------------------
    # 9. 动作冲突（沿用 numpy 实现，不影响效率）
    # -------------------------------------------------
    def deal_conflict_batch(self, actions: torch.Tensor) -> torch.Tensor:
        acts_np = _th2np(actions)
        resolved_np = super().deal_conflict_batch(acts_np)
        return _np2th(resolved_np, dtype=torch.long)

    # 其它：reset_actions / _do_actions / _update_xxx / step
    # -------------------------------------------------
    def reset_actions(self, actions: torch.Tensor) -> torch.Tensor:
        cur_pos_1 = self.cur_pos+1
        same_pos_mask = ((cur_pos_1 == actions) | (actions == 0))
        actions = torch.where(same_pos_mask, cur_pos_1, actions)

        return actions

    def _do_actions(self, ori_actions: np.ndarray | torch.Tensor) -> torch.Tensor:
        actions = _np2th(ori_actions, dtype=torch.long)
        actions = self.reset_actions(actions)
        if not self.use_conflict_model:
            actions = self.deal_conflict_batch(actions)
        self.actions = actions
        self.step_count += 1

        del self.last_pos
        self.last_pos = self.cur_pos
        self.cur_pos = actions - 1

        self.trajectories[..., self.step_count] = _th2np(actions)
        return actions


    # --------- vectorized helpers ----------
    def _update_normal_city_masks(self):
        col_idx = self.cur_pos.view(-1)
        self.mask.view(-1, self.mask.size(-1))[self.batch_salesmen_ar, col_idx] = False
        del col_idx

    def _update_costs(self, last_pos, cur_pos):
        self.costs += self.graph_matrix[self.batch_ar[:, None], last_pos, cur_pos]

    def _update_conflict_count(self, is_stay_up, not_stay_up):
        self.conflict_count_exp += is_stay_up.long()
        update_mask = (self.cur_pos != 0) & not_stay_up
        self.conflict_count[update_mask] = self.conflict_count_exp[update_mask]
        del update_mask

    def _update_stages(self, batch_complete, not_stay_up, last_pos):
        inc_mask = (not_stay_up & ((last_pos == 0) | (self.cur_pos == 0))) \
                   | batch_complete[:, None]
        self.traj_stages = torch.where(inc_mask,
                                       self.traj_stages + 1,
                                       self.traj_stages)
        self.traj_stages = torch.where(
            batch_complete[:, None],
            # |
            # (np.all(actions == 1)),
            self.traj_stages + 1,  # 满足条件时阶段+1
            self.traj_stages  # 否则保持原值
        )

        self.stage_2 = self.traj_stages >= 2
        del inc_mask

    # -------------------------------------------------
    @torch.no_grad()
    def step(self, ori_actions: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
        # 1. 执行动作
        self._do_actions(ori_actions)

        # 2. 更新城市 mask
        self._update_normal_city_masks()

        # 3. 更新成本 / 统计量
        is_stay_up = (self.last_pos == self.cur_pos)
        not_stay_up = ~is_stay_up

        self._update_costs(self.last_pos, self.cur_pos)
        self._update_conflict_count(is_stay_up, not_stay_up)

        # 4. 完成判断 & 阶段
        batch_complete = torch.all(~self.mask, dim=1)  # [B]
        self.mask[batch_complete, 0] = True
        self._update_stages(batch_complete, not_stay_up, self.last_pos)

        del is_stay_up, not_stay_up

        # 5. 奖励
        self._get_reward()

        info = {
            "mask": _th2np(self.mask),
            "salesmen_masks": self._get_salesmen_masks(),
            "masks_in_salesmen": None,
            "dones": _th2np(self.dones) if self.done else None
        }

        if self.done:
            self._update_costs(self.cur_pos, torch.zeros_like(self.cur_pos))
            self._get_reward()
            info.update(
                trajectories=self.trajectories[..., :self.step_count + 2],
                costs=_th2np(self.costs.float()),
                conflict_count=_th2np(self.conflict_count)
            )

        return self._get_salesmen_states(), _th2np(self.rewards.float()), self.done, info

    # -------------------------------------------------
    # 10. 画图 / 其它 util（保持不变）
    # -------------------------------------------------
    def draw(self, graph, cost, trajectory, used_time=0,
             agent_name="agent", draw=True):
        graph_plot = GP()
        one_first = False if agent_name == "or_tools" else True
        return graph_plot.draw_route(
            graph, trajectory, draw=draw,
            title=f"{agent_name}_cost:{cost:.5f}_time:{used_time:.3f}",
            one_first=one_first
        )

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

def test2Env(agent, env1, env2):

    agent_num = 5
    city_num = 50
    batch_size = 32

    g = GG()
    batch_graph = g.generate(batch_size=batch_size, num=city_num)
    batch_graph = GG.augment_xy_data_by_8_fold_numpy(batch_graph)

    env_config = {
        "salesmen": agent_num,
        "cities": city_num,
        "seed": None,
        "mode": 'rand',
        "env_masks_mode": 1,
        "use_conflict_model": True
    }
    states1, env_info1 = env1.reset(env_config, batch_graph)
    states2, env_info2 = env2.reset(env_config, batch_graph)

    assert np.isclose(states1, states2).all()
    def assert_check(d,d2):
        for k,v in d.items():
            if isinstance(v,np.ndarray):
                assert np.isclose(v, d2[k]).all(), f"k:{k}"
            else:
                assert v == d2[k], f"k:{k}"
    assert_check(env_info2,env_info1)

    states = states1
    env_info = env_info1

    salesmen_masks = env_info["salesmen_masks"]
    masks_in_salesmen = env_info["masks_in_salesmen"]
    city_mask = env_info["mask"]
    dones = env_info["dones"]

    agent.reset_graph(batch_graph, agent_num)
    info = {
        "use_conflict_model":True
    }


    done = False
    while not done:
        states_t = _convert_tensor(states, device=agent.device)
        # mask: true :not allow  false:allow

        salesmen_masks_t = _convert_tensor(~salesmen_masks, dtype=torch.bool, device=agent.device)
        if masks_in_salesmen is not None:
            masks_in_salesmen_t = _convert_tensor(~masks_in_salesmen, dtype=torch.bool, device=agent.device)
        else:
            masks_in_salesmen_t = None
        city_mask_t = _convert_tensor(~city_mask, dtype=torch.bool, device=agent.device)
        dones_t = _convert_tensor(dones, dtype=torch.bool, device=agent.device) if dones is not None else None
        info.update({
            "masks_in_salesmen": masks_in_salesmen_t,
            "mask": city_mask_t,
            "dones": dones_t
        })

        acts, act_logp, act_nf, act_entropy = agent.predict(states_t, salesmen_masks_t, info)
        if act_nf is not None:
            states, r, done, env_info = env1.step(act_nf + 1)
        else:
            states, r, done, env_info = env1.step(acts + 1)

        if act_nf is not None:
            states2, r2, done2, env_info2 = env2.step(act_nf + 1)
        else:
            states2, r2, done2, env_info2 = env2.step(acts + 1)

        x = np.isclose(states2, states)
        xx = np.argwhere(~x)

        assert np.isclose(states2, states).all()
        assert np.isclose(r, r2).all()
        assert done2 == done
        assert_check(env_info,env_info2)

        salesmen_masks = env_info["salesmen_masks"]
        masks_in_salesmen = env_info["masks_in_salesmen"]
        city_mask = env_info["mask"]
        dones = env_info["dones"]




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--agent_num", type=int, default=4)
    parser.add_argument("--fixed_agent_num", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad_max_norm", type=float, default=0.5)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=0)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--augment", type=int, default=8)
    parser.add_argument("--repeat_times", type=int, default=1)
    parser.add_argument("--city_nums", type=int, default=5)
    parser.add_argument("--random_city_num", type=bool, default=False)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=999999999)
    parser.add_argument("--env_masks_mode", type=int, default=1,
                        help="0 for only the min cost  not allow back depot; 1 for only the max cost allow back depot")
    parser.add_argument("--eval_interval", type=int, default=400, help="eval  interval")
    parser.add_argument("--use_conflict_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_conflict_model", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--train_actions_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_city_encoder", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--use_agents_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--use_city_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--agents_adv_rate", type=float, default=0.0, help="rate of adv between agents")
    parser.add_argument("--conflict_loss_rate", type=float, default=1.0, help="rate of adv between agents")
    parser.add_argument("--only_one_instance", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--save_model_interval", type=int, default=800, help="save model interval")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--epoch_size", type=int, default=1280000, help="number of instance for each epoch")
    parser.add_argument("--n_epoch", type=int, default=100, help="number of epoch")
    args = parser.parse_args()
    def set_seed(seed=42):
        # 基础库
        random.seed(seed)
        np.random.seed(seed)

        # PyTorch核心设置
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU时
        # # 禁用CUDA不确定算法
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    g = GG()
    batch_graph = g.generate(batch_size=args.batch_size, num=args.city_nums)
    batch_graph = GG.augment_xy_data_by_8_fold_numpy(batch_graph)

    set_seed()
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

    from algorithm.Attn.AgentV1 import Agent as Agent
    from model.AttnModel.Model import Config

    agent = Agent(args, Config)
    agent.load_model(args.agent_id)
    # features_nb, actions_nb, actions_no_conflict_nb, returns_nb, individual_returns_nb, masks_nb, dones_nb = agent.run_batch(
    #     env, graph, env_config["salesmen"], 32)
    from utils.TensorTools import _convert_tensor
    import numpy as np
    import torch
    from envs.GraphGenerator import GraphGenerator as GG
    import time

    start_time = time.time()
    for i in range(100):
        o1 = agent.run_batch_episode(env, batch_graph, args.agent_num, True, info={
            "use_conflict_model": args.use_conflict_model})
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))
    # set_seed()
    # from envs.MTSP.MTSP4 import MTSPEnv as Env4
    # env4 = Env4(env_config)
    #
    # agent = Agent(args, Config)
    # agent.load_model(args.agent_id)
    #
    # start_time = time.time()
    # for i in range(1):
    #     o2 = agent.run_batch_episode(env4, batch_graph, args.agent_num, True, info={
    #         "use_conflict_model": args.use_conflict_model})
    # end_time = time.time()
    # print("--- %s seconds ---" % (end_time - start_time))
    # test2Env(agent,env,env4)
