import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import sys

from sympy.strategies.tree import greedy

sys.path.append("../")
sys.path.append("./")

import torch.multiprocessing as mp
from envs.GraphGenerator import GraphGenerator as GG
from utils.TensorTools import _convert_tensor
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from algorithm.OR_Tools.mtsp import ortools_solve_mtsp
import argparse
from envs.MTSP.MTSP4 import MTSPEnv
from algorithm.DNN5.AgentV1 import AgentV1 as Agent
import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--agent_num", type=int, default=4)
    parser.add_argument("--agent_dim", type=int, default=9)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--grad_max_norm", type=float, default=1.0)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--returns_norm", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=5e-3)
    parser.add_argument("--batch_size", type=float, default=8)
    parser.add_argument("--city_nums", type=int, default=50)
    parser.add_argument("--allow_back", type=bool, default=False)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=50000)
    args = parser.parse_args()

    from envs.GraphGenerator import GraphGenerator as GG
    fig = None
    graphG = GG(args.batch_size, args.city_nums, 2)
    from envs.MTSP.MTSP4 import MTSPEnv

    env = MTSPEnv()
    from algorithm.OR_Tools.mtsp import ortools_solve_mtsp
    from model.n4Model.config import Config
    agent = Agent(args, Config)
    agent.load_model(args.agent_id)
    city_nums = args.city_nums
    agent_nums = args.agent_num

    for i in (range(100_000_000)):
        plt.close()
        eval_graph = graphG.generate(args.batch_size, city_nums)
        ortool_cost_list = []
        st = time.time_ns()
        cost,greedy_trajectory =  agent.eval_episode(env, eval_graph,agent_nums, "greedy")
        ed = time.time_ns()
        greedy_cost = np.mean(cost)
        greedy_time = (ed- st) / 1e9
        print(f"greedy_cost:{greedy_cost},greedy_time:{(ed- st) / 1e9}")

        st = time.time_ns()
        min_sample_costs_list = []
        min_sample_traj_list = []
        batch_eval_graph = eval_graph[np.newaxis,].repeat(32, axis=0)
        batch_eval_graph = batch_eval_graph.reshape(32 * eval_graph.shape[0],eval_graph.shape[1],eval_graph.shape[2])
        cost, trajectory = agent.eval_episode(env, batch_eval_graph,agent_nums, "sample")
        batch_cost = cost.reshape(32, eval_graph.shape[0], -1,)
        min_sample_cost = np.min(batch_cost, axis=0)
        min_sample_cost_mean = np.mean(min_sample_cost)
        ed = time.time_ns()
        sample_time = (ed - st) / 1e9
        print(f"sample_cost:{min_sample_cost_mean},sample_time:{sample_time}")

        ortools_time = 0
        for k in range(args.batch_size):
            ortools_trajectory, ortools_cost, used_time = ortools_solve_mtsp(eval_graph[k:k+1], agent_nums, 10000)
            ortool_cost_list.append(ortools_cost)
            ortools_time  = ortools_time + used_time
        ortools_cost_mean = np.mean(ortool_cost_list)
        print(f"ortools_cost:{ortools_cost_mean},ortools_time:{ortools_time}")

        if args.batch_size == 1:
            greedy_traj = env.compress_adjacent_duplicates_optimized(greedy_trajectory)[0]
            idx = np.argmin(batch_cost, axis=0).item()
            min_sample_traj = trajectory[idx:idx+1]
            sample_traj = env.compress_adjacent_duplicates_optimized(min_sample_traj)[0]
            env.draw_multi(eval_graph[0],[ortools_cost_mean, greedy_cost, min_sample_cost_mean], [ortools_trajectory, greedy_traj, sample_traj],
                           [ortools_time, greedy_time, sample_time],["or_tools", "greedy", "sample"])

        traj = env.compress_adjacent_duplicates_optimized(trajectory)
        gap = ((greedy_cost - ortools_cost_mean) / ortools_cost_mean).item()*100