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
    parser.add_argument("--agent_num", type=int, default=100)
    parser.add_argument("--fixed_agent_num", type=bool, default=False)
    parser.add_argument("--agent_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_max_norm", type=float, default=1)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=1e-2)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--city_nums", type=int, default=2000)
    parser.add_argument("--random_city_num", type=bool, default=True)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=120000)
    parser.add_argument("--env_masks_mode", type=int, default=4,
                        help="0 for only the min cost  not allow back depot; 1 for only the max cost allow back depot")
    parser.add_argument("--eval_interval", type=int, default=100, help="eval  interval")
    parser.add_argument("--use_conflict_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_conflict_model", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--train_actions_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_city_encoder", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--use_agents_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--use_city_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--agents_adv_rate", type=float, default=0.0, help="rate of adv between agents")
    parser.add_argument("--conflict_loss_rate", type=float, default=0.5 + 0.5, help="rate of adv between agents")
    parser.add_argument("--only_one_instance", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--save_model_interval", type=int, default=10000, help="save model interval")
    parser.add_argument("--seed", type=int, default=528, help="random seed")
    parser.add_argument("--draw", type=bool, default=True, help="whether to draw result")

    args = parser.parse_args()

    from envs.GraphGenerator import GraphGenerator as GG
    fig = None
    graphG = GG(args.batch_size, args.city_nums, 2)
    from envs.MTSP.MTSP5 import MTSPEnv

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
        # ortools_cost, ortools_traj, ortools_time = EvalTools.EvalOrTools(graph, agent_nums)
        LKH_cost, LKH_traj, LKH_time = EvalTools.EvalLKH3(graph, agent_nums)

        if args.batch_size == 1:
            env.draw_multi(
                graph,
                [greedy_cost, no_conflict_greedy_cost, sample_cost, LKH_cost],# ortools_cost],
                [greedy_traj, no_conflict_greedy_traj, sample_traj, LKH_traj],# ortools_traj],
                [greedy_time, no_conflict_greedy_time, sample_time, LKH_time],# ortools_time],
                ["greedy","no_conflict_greedy", "sample", "LKH"],#"or_tools"],
            )