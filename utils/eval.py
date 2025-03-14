import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from utils.TspInstanceFileTool import *
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
    parser.add_argument("--agent_num", type=int, default=5)
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
    parser.add_argument("--batch_size", type=float, default=1)
    parser.add_argument("--city_nums", type=int, default=30)
    parser.add_argument("--allow_back", type=bool, default=False)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=100, help="eval  interval")
    parser.add_argument("--env_masks_mode", type=int, default=1, help="0 for only the min cost  not allow back depot; 1 for only the max cost allow back depot")
    parser.add_argument("--use_conflict_model", type=bool, default=True, help="0:not use;1:use")

    args = parser.parse_args()

    from envs.GraphGenerator import GraphGenerator as GG
    fig = None
    graphG = GG(args.batch_size, args.city_nums, 2)
    from envs.MTSP.MTSP4 import MTSPEnv

    env = MTSPEnv({"env_masks_mode":args.env_masks_mode})
    from algorithm.OR_Tools.mtsp import ortools_solve_mtsp
    from model.n4Model.config import Config
    agent = Agent(args, Config)
    agent.load_model(args.agent_id)
    city_nums = args.city_nums
    agent_nums = args.agent_num
    from EvalTools import EvalTools
    for i in (range(100_000_000)):
        graph = graphG.generate()

        greedy_cost, greedy_traj, greedy_time = EvalTools.EvalGreedy(graph, agent_nums, agent, env)
        no_conflict_greedy_cost, no_conflict_greedy_traj, no_conflict_greedy_time=EvalTools.EvalGreedy(graph, agent_nums, agent, env,{"use_conflict_model":False})
        sample_cost, sample_traj ,sample_time = EvalTools.EvalSample(graph, agent_nums, agent, env)
        ortools_cost, ortools_traj, ortools_time = EvalTools.EvalOrTools(graph, agent_nums)
        LKH_cost, LKH_traj, LKH_time = EvalTools.EvalLKH3(graph, agent_nums)

        if args.batch_size == 1:
            env.draw_multi(
                graph,
                [greedy_cost, no_conflict_greedy_cost, sample_cost, ortools_cost, LKH_cost],
                [greedy_traj, no_conflict_greedy_traj, sample_traj, ortools_traj, LKH_traj],
                [greedy_time, no_conflict_greedy_time, sample_time, ortools_time, LKH_time],
                ["greedy","no_conflict_greedy", "sample","or_tools", "LKH"]
            )