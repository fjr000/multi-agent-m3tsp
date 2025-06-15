# ================== MTSPEnvTorch.py ==================
"""
Mixed-precision 版 MTSP 环境
接口与旧版保持一致，内部主要张量均使用 float16 以降低显存
"""
from __future__ import annotations
import argparse, math
from typing import Dict, Tuple, List

import numpy as np          # 仅在出/入口使用
import torch, torch.nn.functional as F


# -----------------------------------------------------
# 公共工具
# -----------------------------------------------------
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FP_DTYPE = torch.float32        # 存储用
ACC_DTYPE = torch.float32       # 计算用(距 离/代价)


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
        self.stage_2              : torch.Tensor | None = None
        self.cur_pos              : torch.Tensor | None = None
        self.cities               = 50
        self.salesmen             = 5
        self.seed                 = None
        self.mode                 = "rand"
        self.problem_size         = 1
        self.env_masks_mode       = 0
        self.use_conflict_model   = False

        # 维度常数
        self.dim                  = 13

        # 其余运行时变量
        self.step_count           = 0
        self.graph                : torch.Tensor | None = None
        self.graph_matrix         : torch.Tensor | None = None
        self.trajectories         : torch.Tensor | None = None
        self.costs                : torch.Tensor | None = None
        self.mask                 : torch.Tensor | None = None
        self.traj_stages          : torch.Tensor | None = None
        self.salesmen_mask        : torch.Tensor | None = None
        self.actions              : torch.Tensor | None = None
        self.ori_actions          : torch.Tensor | None = None
        self.conflict_count_exp   : torch.Tensor | None = None
        self.conflict_count       : torch.Tensor | None = None
        self.dones                : torch.Tensor | None = None
        self.states               : torch.Tensor | None = None
        self.batch_ar             : torch.Tensor | None = None
        self.batch_salesmen_ar    : torch.Tensor | None = None
        self.rewards              : torch.Tensor | None = None
        self.done                 : bool = False
        self._min_distance        : torch.Tensor | None = None

        if config is not None:
            self._parse_config(config)

    # -------------------------------------------------
    # 2. 解析外部配置
    # -------------------------------------------------
    def _parse_config(self, config: Dict):
        self.cities             = config.get("cities", self.cities)
        self.salesmen           = config.get("salesmen", self.salesmen)
        self.seed               = config.get("seed", self.seed)
        self.mode               = config.get("mode", self.mode)
        self.problem_size       = config.get("N_aug", self.problem_size)
        self.env_masks_mode     = config.get("env_masks_mode", self.env_masks_mode)
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
            tg = _np2th(graph)            # 已转半精度
            if tg.ndim == 2:
                tg = tg.unsqueeze(0)
            elif tg.ndim != 3:
                raise ValueError("graph must be [N,2] or [B,N,2]")
            self.graph = tg                # [B,N,2]
            self.problem_size = self.graph.size(0)

        elif self.graph is None or self.mode == "rand":
            g_np  = self.GG.generate(self.problem_size, self.cities, 2)
            self.graph = _np2th(g_np)      # [B,N,2], 半精度

        # 距离矩阵 [B,N,N] —— 生成后直接 cast
        self.graph_matrix = torch.cdist(self.graph, self.graph)

        # -------- 3.2 各种运行时张量 ----------
        B, A, N = self.problem_size, self.salesmen, self.cities
        self.step_count   = 0

        self.trajectories = np.ones((B, A, N + 1), dtype=np.int_)
        self.cur_pos      = _np2th(self.trajectories[..., 0] - 1)             # [B,A]

        self.costs        = torch.zeros(B, A, dtype=ACC_DTYPE, device=device)
        self.mask         = torch.ones (B, N, dtype=torch.bool, device=device)
        self.mask[:, 0]   = False                                     # depot 不可选

        self.traj_stages  = torch.zeros(B, A, dtype=torch.long, device=device)
        self.stage_2      = torch.zeros_like(self.traj_stages, dtype=torch.bool)

        self.batch_ar          = torch.arange(B, device=device)
        self.batch_salesmen_ar = self.batch_ar.repeat_interleave(A)   # [B*A]

        self.conflict_count_exp = torch.zeros(B, A, dtype=torch.long,  device=device)
        self.conflict_count     = torch.zeros_like(self.conflict_count_exp)

        # min distance (排除 0)
        gm_tmp = torch.where(
            torch.isclose(self.graph_matrix, torch.tensor(0., device=device, dtype=FP_DTYPE)),
            torch.tensor(float('inf'), device=device, dtype=FP_DTYPE),
            self.graph_matrix
        )
        self._min_distance = torch.min(gm_tmp.view(B, -1), dim=-1).values   # [B]

    # -------------------------------------------------
    # 4. 计算智能体 state
    # -------------------------------------------------
    @torch.no_grad()
    def _get_salesmen_states(self) -> np.ndarray:
        B, A, N = self.problem_size, self.salesmen, self.cities
        depot_idx = 0
        dev = self.graph_matrix.device
        # ------- 先把必要的 FP16 张量升到 FP32 -------
        gmat32 = self.graph_matrix.float()  # [B,N,N] fp32
        costs32 = self.costs  # 本身已 fp32
        cur_pos = self.cur_pos
        mask_ban = self.mask
        batch_idx = self.batch_ar[:, None]  # [B,1]
        city_mask = mask_ban[:, None, :]  # [B,1,N]  True=未访问
        cur_dists = gmat32[batch_idx, cur_pos]  # [B,A,N] fp32
        remain_cts = torch.sum(mask_ban, dim=1, keepdim=True).float()  # fp32
        # ---------- 各距离 / 统计 ----------
        max_cost = torch.max(costs32, dim=1, keepdim=True).values
        min_cost = torch.min(costs32, dim=1, keepdim=True).values
        depot_dist = gmat32[batch_idx, cur_pos, 0]  # [B,A]
        neg_inf = torch.tensor(-float("inf"), device=dev, dtype=ACC_DTYPE)
        pos_inf = torch.tensor(float("inf"), device=dev, dtype=ACC_DTYPE)
        masked_for_max = torch.where(city_mask, cur_dists, neg_inf)
        masked_for_min = torch.where(city_mask, cur_dists, pos_inf)
        max_dists = torch.max(masked_for_max, dim=-1).values
        min_dists = torch.min(masked_for_min, dim=-1).values
        max_dists = torch.where(torch.isinf(max_dists),
                                torch.zeros_like(max_dists), max_dists)
        min_dists = torch.where(torch.isinf(min_dists),
                                torch.zeros_like(min_dists), min_dists)
        depot_line = gmat32[:, 0:1, :]  # [B,1,N]
        two_step = cur_dists + depot_line
        masked_two_max = torch.where(city_mask, two_step, neg_inf)
        masked_two_min = torch.where(city_mask, two_step, pos_inf)
        max_two = torch.max(masked_two_max, dim=-1).values
        min_two = torch.min(masked_two_min, dim=-1).values
        max_two = torch.where(torch.isinf(max_two), torch.zeros_like(max_two), max_two)
        min_two = torch.where(torch.isinf(min_two), torch.zeros_like(min_two), min_two)
        max_max_dep = torch.max(costs32 + max_two, dim=1, keepdim=True).values
        progress = 1.0 - remain_cts / (N - 1)  # fp32
        # ------------- 拼装并再降精度 -------------
        scale = max_max_dep
        allow_stay = self.salesmen_mask[
            batch_idx,
            torch.arange(A, device=dev)[None, :],
            cur_pos
        ]
        states32 = torch.empty(B, A, self.dim, dtype=ACC_DTYPE, device=dev)
        states32[..., 0] = depot_idx
        states32[..., 1] = cur_pos
        states32[..., 2] = costs32 / scale
        states32[..., 3] = depot_dist / scale
        states32[..., 4] = max_dists / scale
        states32[..., 5] = min_dists / scale
        states32[..., 6] = max_two / scale
        states32[..., 7] = min_two / scale
        states32[..., 8] = allow_stay.float()
        states32[..., 9] = max_cost / scale
        states32[..., 10] = min_cost / scale
        states32[..., 11] = max_max_dep / scale
        states32[..., 12] = progress
        self.states = states32.to(FP_DTYPE)  # ↓ 存半精度
        return self.states.float().cpu().numpy()  # 返 float32

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

        repeat_masks   = self.mask[:, None, :].repeat(1, A, 1)   # [B,A,N]
        active_agents  = self.traj_stages == 1                   # [B,A]
        batch_idx1d    = torch.nonzero(torch.sum(active_agents, dim=1) > 1).squeeze(1)
        batch_idx      = batch_idx1d[:, None]                    # [K,1]

        if self.env_masks_mode in (0, 2):
            dis_depot = self.graph_matrix[batch_idx, self.cur_pos[batch_idx1d], 0] \
                        if self.env_masks_mode >= 2 else 0.
            expect_dis = self.costs[batch_idx1d] + dis_depot
            masked_cost = torch.where(active_agents[batch_idx1d], expect_dis,
                                      torch.tensor(float('inf'), device=device, dtype=FP_DTYPE))
            min_cost_idx = torch.argmin(masked_cost, dim=-1)
            repeat_masks[batch_idx, :, 0] = True
            repeat_masks[batch_idx, min_cost_idx[:, None], 0] = False
            # allow stay
            repeat_masks[batch_idx, torch.arange(A, device=device)[None, :],
                         self.cur_pos[batch_idx1d]] = active_agents[batch_idx1d]
            repeat_masks[batch_idx, min_cost_idx[:, None],
                         self.cur_pos[batch_idx1d, min_cost_idx]] = False

        elif self.env_masks_mode in (1, 3):
            dis_depot = self.graph_matrix[batch_idx, self.cur_pos[batch_idx1d], 0] \
                        if self.env_masks_mode >= 3 else 0.
            expect_dis = self.costs[batch_idx1d] + dis_depot
            masked_cost = torch.where(active_agents[batch_idx1d], expect_dis,
                                      torch.tensor(float('-inf'), device=device, dtype=FP_DTYPE))
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
            "graph"            : _th2np(self.graph.float()),      # float32 给外界
            "salesmen"         : self.salesmen,
            "cities"           : self.cities,
            "mask"             : _th2np(self.mask),
            "salesmen_masks"   : self._get_salesmen_masks(),
            "masks_in_salesmen": None,
            "min_distance"     : _th2np(self._min_distance.float()),
            "dones"            : None
        }
        return self._get_salesmen_states(), env_info

    # -------------------------------------------------
    # 8. reward / done 判断
    # -------------------------------------------------
    def _get_reward(self):
        self.dones   = torch.all(self.stage_2, dim=1)         # [B]
        self.done    = bool(torch.all(self.dones))
        max_cost     = torch.max(self.costs, dim=1).values     # [B]
        self.rewards = torch.where(self.dones, -max_cost, torch.zeros_like(max_cost))
        return self.rewards

    # -------------------------------------------------
    # 9. 动作冲突（沿用 numpy 实现，不影响效率）
    # -------------------------------------------------
    def deal_conflict_batch(self, actions: torch.Tensor) -> torch.Tensor:
        acts_np      = _th2np(actions)
        resolved_np  = super().deal_conflict_batch(acts_np)
        return _np2th(resolved_np, dtype=torch.long)

    # 其它：reset_actions / _do_actions / _update_xxx / step
    # -------------------------------------------------
    def reset_actions(self, actions: torch.Tensor) -> torch.Tensor:
        same_pos_mask = ((self.cur_pos == actions) | (actions == 0))
        actions = torch.where(same_pos_mask, self.cur_pos, actions)

        return actions

    def _do_actions(self, ori_actions: np.ndarray | torch.Tensor) -> torch.Tensor:
        actions = _np2th(ori_actions, dtype=torch.long)
        actions = self.reset_actions(actions)
        if not self.use_conflict_model:
            actions = self.deal_conflict_batch(actions)
        self.actions     = actions
        self.step_count += 1
        self.trajectories[..., self.step_count] = _th2np(actions)
        return actions

    # --------- vectorized helpers ----------
    def _update_normal_city_masks(self, actions: torch.Tensor):
        col_idx = (actions - 1).view(-1)
        self.mask.view(-1, self.mask.size(-1))[self.batch_salesmen_ar, col_idx] = False

    def _update_costs(self, last_pos, cur_pos):
        self.costs += self.graph_matrix[self.batch_ar[:, None], last_pos, cur_pos]

    def _update_conflict_count(self, is_stay_up, not_stay_up):
        self.conflict_count_exp += is_stay_up.long()
        update_mask = (self.cur_pos != 0) & not_stay_up
        self.conflict_count[update_mask] = self.conflict_count_exp[update_mask]

    def _update_stages(self, batch_complete, not_stay_up, last_pos):
        inc_mask = (not_stay_up & ((last_pos == 0) | (self.cur_pos == 0))) \
                   | batch_complete[:, None]
        self.traj_stages = torch.where(inc_mask,
                                       self.traj_stages + 1,
                                       self.traj_stages)
        self.stage_2 = self.traj_stages >= 2

    # -------------------------------------------------
    @torch.no_grad()
    def step(self, ori_actions: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
        # 1. 执行动作
        actions   = self._do_actions(ori_actions)

        # 2. 更新城市 mask
        self._update_normal_city_masks(actions)

        # 3. 更新成本 / 统计量
        last_pos  = _np2th(self.trajectories[..., self.step_count - 1] - 1)
        self.cur_pos = _np2th(self.trajectories[..., self.step_count] - 1)
        is_stay_up  = last_pos == self.cur_pos
        not_stay_up = ~is_stay_up

        self._update_costs(last_pos, self.cur_pos)
        self._update_conflict_count(is_stay_up, not_stay_up)

        # 4. 完成判断 & 阶段
        batch_complete = torch.all(~self.mask, dim=1)           # [B]
        self.mask[batch_complete, 0] = True
        self._update_stages(batch_complete, not_stay_up, last_pos)

        # 5. 奖励
        self._get_reward()

        info = {
            "mask"              : _th2np(self.mask),
            "salesmen_masks"    : self._get_salesmen_masks(),
            "masks_in_salesmen" : None,
            "dones"             : _th2np(self.dones) if self.done else None
        }

        if self.done:
            self._update_costs(self.cur_pos, torch.zeros_like(self.cur_pos))
            self._get_reward()
            info.update(
                trajectories   = self.trajectories[..., :self.step_count + 2],
                costs          = _th2np(self.costs.float()),
                conflict_count = _th2np(self.conflict_count)
            )

        return self._get_salesmen_states(), _th2np(self.rewards.float()), self.done, info

    # -------------------------------------------------
    # 10. 画图 / 其它 util（保持不变）
    # -------------------------------------------------
    def draw(self, graph, cost, trajectory, used_time=0,
             agent_name="agent", draw=True):
        graph_plot = GP()
        one_first  = False if agent_name == "or_tools" else True
        return graph_plot.draw_route(
            graph, trajectory, draw=draw,
            title=f"{agent_name}_cost:{cost:.5f}_time:{used_time:.3f}",
            one_first=one_first
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--agent_num", type=int, default=10)
    parser.add_argument("--fixed_agent_num", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad_max_norm", type=float, default=0.5)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=0)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--augment", type=int, default=8)
    parser.add_argument("--repeat_times", type=int, default=1)
    parser.add_argument("--city_nums", type=int, default=50)
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

    from algorithm.Attn.AgentV7 import Agent as Agent
    from model.AttnModel.ModelV7 import Config

    agent = Agent(args, Config)
    agent.load_model(args.agent_id)
    # features_nb, actions_nb, actions_no_conflict_nb, returns_nb, individual_returns_nb, masks_nb, dones_nb = agent.run_batch(
    #     env, graph, env_config["salesmen"], 32)
    from utils.TensorTools import _convert_tensor
    import numpy as np
    import torch
    from envs.GraphGenerator import GraphGenerator as GG

    def set_seed(seed=42):
        # 基础库
        np.random.seed(seed)

        # PyTorch核心设置
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU时
        # # 禁用CUDA不确定算法
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    set_seed()

    g = GG()
    batch_graph = g.generate(batch_size=args.batch_size, num=args.city_nums)
    batch_graph = GG.augment_xy_data_by_8_fold_numpy(batch_graph)
    states, info = env.reset(graph=batch_graph)
    salesmen_masks = info["salesmen_masks"]
    agent.reset_graph(batch_graph, 3)
    done = False
    import time
    start_time = time.time()
    for i in range(1):
        agent.run_batch_episode(env, batch_graph, args.agent_num, False, info={
            "use_conflict_model": args.use_conflict_model})
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))
