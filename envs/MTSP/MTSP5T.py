import argparse
import copy
import math
import time

import numpy as np
from typing import Tuple, List, Dict
import sys

from sympy import timed

from envs.MTSP.MTSP4T import MTSPEnv as Env

sys.path.append("../")
from envs.MTSP.Config import Config
from envs.GraphGenerator import GraphGenerator as GG
from utils.GraphPlot import GraphPlot as GP
from model.NNN.RandomAgent import RandomAgent
import torch.nn.functional as F
import torch


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

    @torch.no_grad()
    def _get_salesmen_masks(self) -> np.ndarray:
        """
        生成 shape=[B,A,N] 的布尔张量 salesmen_mask，
        Torch-only 实现，兼容缺少 torch.nan* API 的旧版本。
        """
        # ------------ 一些通用变量 ------------
        B, N = self.mask.size()  # [B,N]
        A = self.salesmen
        dev = self.mask.device
        inf = torch.tensor(float('inf'), device=dev)
        ninf = torch.tensor(-float('inf'), device=dev)
        nan = torch.tensor(float('nan'), device=dev)

        repeat_masks = self.mask[:, None, :].repeat(1, A, 1).clone()  # bool

        batch_ar = self.batch_ar  # [B]
        cur_pos = self.cur_pos  # [B,A]
        costs = self.costs  # [B,A]
        gmat = self.graph_matrix  # [B,N,N]
        mode = self.env_masks_mode

        # ----------- 兼容函数 -----------
        def _nanmean(x, dim, keepdim=False):
            """mean 忽略 nan"""
            mask = x.isnan()
            s = torch.where(mask, torch.zeros_like(x), x).sum(dim=dim, keepdim=keepdim)
            cnt = (~mask).sum(dim=dim, keepdim=keepdim).clamp(min=1)
            return s / cnt

        def _nanmin(x, dim):
            """min 忽略 nan"""
            return torch.where(x.isnan(), inf, x).min(dim=dim).values

        def _nanmax(x, dim):
            """max 忽略 nan"""
            return torch.where(x.isnan(), ninf, x).max(dim=dim).values

        # ============ mode 0 / 1 / 2 / 3 ============
        if mode in (0, 1, 2, 3):
            active_agents = (self.traj_stages == 1)  # [B,A]
            sel_batch = torch.nonzero(active_agents.sum(-1) > 1, as_tuple=False).squeeze(1)
            if sel_batch.numel():
                K = sel_batch.size(0)
                active_sub = active_agents[sel_batch]  # [K,A]
                cur_pos_sub = cur_pos[sel_batch]  # [K,A]
                costs_sub = costs[sel_batch]  # [K,A]

                # ------- 最小 / 最大成本选择 -------
                if mode in (0, 2):  # 最小
                    if mode == 2:  # 带返回仓库距离
                        idx_b = sel_batch.view(-1, 1).expand(-1, A)
                        dis_dep = gmat[idx_b, cur_pos_sub, 0]
                        costs_eff = costs_sub + dis_dep
                    else:
                        costs_eff = costs_sub
                    masked_cost = torch.where(active_sub, costs_eff, inf)
                    sel_idx = masked_cost.argmin(dim=-1)  # [K]
                    allow_flag = True
                else:  # 最大
                    if mode == 3:
                        idx_b = sel_batch.view(-1, 1).expand(-1, A)
                        dis_dep = gmat[idx_b, cur_pos_sub, 0]
                        costs_eff = costs_sub + dis_dep
                    else:
                        costs_eff = costs_sub
                    masked_cost = torch.where(active_sub, costs_eff, ninf)
                    sel_idx = masked_cost.argmax(dim=-1)  # [K]
                    allow_flag = False

                # 批量写入
                b_flat = sel_batch.unsqueeze(1).expand(-1, A).reshape(-1)
                a_flat = torch.arange(A, device=dev).expand(K, A).reshape(-1)
                pos_flat = cur_pos_sub.reshape(-1)
                repeat_masks[b_flat, a_flat, pos_flat] = active_sub.reshape(-1) ^ allow_flag
                pos_sel = cur_pos[sel_batch, sel_idx]
                repeat_masks[sel_batch, sel_idx, pos_sel] = allow_flag

        # ============ mode 4 / 5 / 6 / 7 / 8 ============
        if mode in (4, 5, 6, 7, 8):
            active_agents = (self.traj_stages <= 1)  # [B,A]
            sel_batch = torch.nonzero(active_agents.sum(-1) > 1, as_tuple=False).squeeze(1)
            if sel_batch.numel():
                K = sel_batch.size(0)
                cur_pos_sub = cur_pos[sel_batch]  # [K,A]
                costs_sub = costs[sel_batch]  # [K,A]

                # 预先算期望距离
                idx_b = sel_batch.view(-1, 1).expand(-1, A)
                dis_dep = gmat[idx_b, cur_pos_sub, 0]  # [K,A]
                exp_dis = costs_sub + dis_dep  # [K,A]

                if mode == 4:
                    masked_cost = torch.where(active_agents[sel_batch], exp_dis, nan)
                    mean = _nanmean(masked_cost, 1, keepdim=True)  # [K,1]
                    stay_mask = masked_cost > mean  # [K,A]

                    # 布尔化写入
                    b_flat = sel_batch.unsqueeze(1).expand(-1, A).reshape(-1)
                    a_flat = torch.arange(A, device=dev).expand(K, A).reshape(-1)
                    pos_flat = cur_pos_sub.reshape(-1)
                    repeat_masks[b_flat, a_flat, pos_flat] = stay_mask.reshape(-1)

                    # 禁止最小的
                    inf_masked = torch.where(active_agents[sel_batch], exp_dis, inf)
                    min_idx = inf_masked.argmin(dim=-1)
                    pos_min = cur_pos[sel_batch, min_idx]
                    repeat_masks[sel_batch, min_idx, pos_min] = False

                else:  # mode 5 / 6 / 7 / 8 共用
                    N_city = gmat.size(-1)
                    cur_pos_exp = cur_pos_sub.unsqueeze(-1).expand(-1, -1, N_city)
                    gmat_sel = gmat[sel_batch]  # [K,N,N]
                    sel_dists = gmat_sel.gather(1, cur_pos_exp)  # [K,A,N]
                    depot_line = gmat_sel[:, 0:1, :]  # [K,1,N]
                    trip_cost = costs_sub.unsqueeze(-1) + sel_dists + depot_line  # [K,A,N]

                    city_mask = self.mask[sel_batch].unsqueeze(1)  # [K,1,N]

                    if mode in (5, 6):
                        masked = torch.where(city_mask, trip_cost, inf)
                        min_dist = masked.min(dim=2).values  # [K,A]
                    else:  # 7 / 8
                        masked = torch.where(city_mask, trip_cost, nan)
                        min_dist = _nanmin(masked, 2)  # [K,A]
                        max_dist = _nanmax(masked, 2)  # [K,A]

                    max_exp = exp_dis.max(dim=-1, keepdim=True)[0]  # [K,1]

                    if mode == 5:
                        min_min = min_dist.min(dim=1, keepdim=True)[0]
                        allow_stay = (min_dist >= max_exp) & (min_dist != min_min)
                    elif mode == 6:
                        min_min = min_dist.min(dim=1, keepdim=True)[0]
                        allow_stay = (min_dist != min_min)
                    elif mode == 7:
                        min_min = min_dist.min(dim=1, keepdim=True)[0]
                        allow_stay = (max_dist >= max_exp) & (min_dist != min_min)
                    else:  # mode 8
                        min_max = max_dist.min(dim=1, keepdim=True)[0]
                        allow_stay = (max_dist >= max_exp) & (max_dist != min_max)

                    # 写入 allow_stay
                    b_flat = sel_batch.unsqueeze(1).expand(-1, A).reshape(-1)
                    a_flat = torch.arange(A, device=dev).expand(K, A).reshape(-1)
                    pos_flat = cur_pos_sub.reshape(-1)
                    repeat_masks[b_flat, a_flat, pos_flat] = allow_stay.reshape(-1)

        # ============ traj stage ≥2 处理 ============
        if self.stage_2.any():
            repeat_masks[self.stage_2] = False
            b_idx, a_idx = torch.nonzero(self.stage_2, as_tuple=True)
            repeat_masks[b_idx, a_idx, 0] = True

        self.salesmen_mask = repeat_masks
        return self.salesmen_mask.detach().cpu().numpy()


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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--augment", type=int, default=8)
    parser.add_argument("--repeat_times", type=int, default=1)
    parser.add_argument("--city_nums", type=int, default=50)
    parser.add_argument("--random_city_num", type=bool, default=False)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=999999999)
    parser.add_argument("--env_masks_mode", type=int, default=7,
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
    for i in range(100):
        agent.run_batch_episode(env, batch_graph, args.agent_num, False, info={
            "use_conflict_model": args.use_conflict_model})
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))



