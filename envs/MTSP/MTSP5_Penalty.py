import argparse
import copy
import math
import time

import numpy as np
from typing import Tuple, List, Dict, override
import sys
from envs.MTSP.MTSP5 import MTSPEnv as Env

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

    @override
    def step(self, ori_actions: np.ndarray):
        s,r,d,info = super(MTSPEnv, self).step(ori_actions)

        if self.done:
            trajs = info['trajectories']
            last_non_one, last_indices = self.get_last_non_one_vectorized(trajs)
            cur_pos = trajs[...,1:]
            last_pos = trajs[...,:-1]
            penalty = np.zeros_like(trajs)
            # penalty = ((cur_pos != 1) & (cur_pos == last_pos) & (cur_pos != last_non_one[...,np.newaxis])).astype(np.int_)
            # sum_penalty = penalty.sum(axis=-1)
            for i in range(1,trajs.shape[-1]):
                penalty[...,i] = (trajs[...,i] == trajs[...,i-1]) & (trajs[...,i] !=last_non_one) & (i < last_indices)
            penalty = penalty * self._min_distance[:,None,None] * trajs.shape[1] / self.problem_size
            info.update({'penalty':penalty})
            # sum_penalty = penalty.sum(axis=-1)
            # check = np.all(sum_penalty == info['conflict_count'])
            # check_pos = np.argwhere(sum_penalty != info['conflict_count'])

        return s, r, d, info

    def get_last_non_one_vectorized(self, x):
        # x.shape = [B, A, T]
        B,A,T = x.shape
        mask = (x != 1)  # shape: [B, A, T]

        # 反转最后维度，找到第一个非1的位置（即原序列最后一个非1）
        flipped_mask = np.flip(mask, axis=-1)

        # 找到反转后第一个 True 的位置，即原序列中最后一个非1的位置
        last_indices = T - 1 - np.argmax(flipped_mask, axis=-1)

        # 处理全为1的情况（argmax会在0位置返回False），这里将没有非1的设为T
        all_ones = ~np.any(mask, axis=-1)
        last_indices[all_ones] = 0  # 或者设为 None / NaN 等

        # 使用高级索引提取对应值
        B, A, T = x.shape
        result = x[np.arange(B)[:, None, None],
        np.arange(A)[None, :, None],
        last_indices[..., None]]

        return result.reshape(B, A), last_indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_worker", type=int, default=12)
    parser.add_argument("--agent_num", type=int, default=2)
    parser.add_argument("--fixed_agent_num", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_max_norm", type=float, default=0.5)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=1e-3)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--augment", type=int, default=8)
    parser.add_argument("--repeat_times", type=int, default=4)
    parser.add_argument("--city_nums", type=int, default=10)
    parser.add_argument("--random_city_num", type=bool, default=False)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=999999999)
    parser.add_argument("--env_masks_mode", type=int, default=7,
                        help="0 for only the min cost  not allow back depot; 1 for only the max cost allow back depot")
    parser.add_argument("--eval_interval", type=int, default=800, help="eval  interval")
    parser.add_argument("--use_conflict_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_conflict_model", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--train_actions_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_city_encoder", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--use_agents_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--use_city_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--agents_adv_rate", type=float, default=0.0, help="rate of adv between agents")
    parser.add_argument("--conflict_loss_rate", type=float, default=1.0, help="rate of adv between agents")
    parser.add_argument("--only_one_instance", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--save_model_interval", type=int, default=1600, help="save model interval")
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

    g = GG()


    def repeat_batch_adjacent(arr: np.ndarray, K: int) -> np.ndarray:
        B, N, _ = arr.shape
        # 扩展维度 -> [B, 1, N, 2]
        expanded = arr[:, None]  # 等价于 np.expand_dims(arr, axis=1)
        # 重复 K 次 -> [B, K, N, 2]
        repeated = np.repeat(expanded, repeats=K, axis=1)
        # reshape 成 [B*K, N, 2]
        result = repeated.reshape(-1, N, 2)
        return result
    batch_graph = g.generate(batch_size=args.batch_size, num=args.city_nums)
    graph_8 = GG.augment_xy_data_by_8_fold_numpy(batch_graph)
    batch_graph = graph_8 = repeat_batch_adjacent(graph_8, args.repeat_times)

    states, info = env.reset(graph=batch_graph)
    salesmen_masks = info["salesmen_masks"]
    st = time.time_ns()
    agent.reset_graph(batch_graph, args.agent_num)
    done = False
    for i in range(100):
        output = agent.run_batch_episode(env, batch_graph, args.agent_num, False, info={
            "use_conflict_model": args.use_conflict_model})
        pass
    ed = time.time_ns()
    print((ed - st) / 1e9)
