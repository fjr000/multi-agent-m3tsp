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
        A = self.salesmen
        repeat_masks = self.mask.unsqueeze(1).repeat(1, A, 1)  # 结果形状 (B, A, L)

        if self.env_masks_mode == 7:
            active_agents = self.traj_stages <= 1  # [B,A]
            batch_indices_1d = self.batch_ar[torch.sum(active_agents, dim=-1) > 1]  # [B,]
            batch_indices = batch_indices_1d[:, None]  # [B,A]

            cur_costs = self.costs[batch_indices_1d]  # [B,A]
            cur_pos = self.cur_pos[batch_indices_1d]  # [B,A]
            dis_depot = self.graph_matrix[batch_indices, cur_pos, 0]  # [B,A]
            expect_dis = cur_costs + dis_depot  # [B,A]
            max_expect_dis = torch.max(expect_dis, dim=-1, keepdim=True)[0]

            selected_dists = self.graph_matrix[batch_indices, cur_pos]  # [B,A,N]
            each_depot_dist = self.graph_matrix[batch_indices_1d, 0:1, :]  # Depot to all cities [B,1,N]
            selected_dists_depot = cur_costs[..., None] + selected_dists + each_depot_dist  # [B,A,N]
            masked_dist_depot_inf = torch.where(self.mask[batch_indices_1d, None, :], selected_dists_depot, torch.inf)
            masked_dist_depot_ninf = torch.where(self.mask[batch_indices_1d, None, :], selected_dists_depot, -torch.inf)
            min_dist_depot = torch.min(masked_dist_depot_inf, dim=2)[0]  # [B,A]
            max_dist_depot = torch.max(masked_dist_depot_ninf, dim=2)[0]  # [B,A]
            min_min_dist_depot = torch.min(min_dist_depot, dim=1, keepdim=True)[0]
            allow_stay = ((max_dist_depot >= max_expect_dis) & (min_dist_depot != min_min_dist_depot))

            repeat_masks[batch_indices, torch.arange(A)[None,], cur_pos] = allow_stay
        else:
            raise NotImplementedError


        # ============ traj stage ≥2 处理 ============
        if self.stage_2.any():
            repeat_masks[self.stage_2] = False
            b_idx, a_idx = torch.nonzero(self.stage_2, as_tuple=True)
            repeat_masks[b_idx, a_idx, 0] = True
        #
        self.salesmen_mask = repeat_masks
        # if torch.all(~repeat_masks,dim=2).any():
        #     x = ~repeat_masks
        #     xxx = torch.all(x,dim=2)
        #     xxxxx= torch.argwhere(xxx)
        #     pass

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



