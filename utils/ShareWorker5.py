import sys
import time

import numpy as np
import sys

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
    parser.add_argument("--agent_num", type=int, default=1)
    parser.add_argument("--agent_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_max_norm", type=float, default=10)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--returns_norm", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=float, default=64)
    parser.add_argument("--city_nums", type=int, default=40)
    parser.add_argument("--allow_back", type=bool, default=False)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=0)
    args = parser.parse_args()

    from envs.GraphGenerator import GraphGenerator as GG

    graphG = GG(args.batch_size, args.city_nums, 2)
    from envs.MTSP.MTSP4 import MTSPEnv

    env = MTSPEnv()
    from algorithm.OR_Tools.mtsp import ortools_solve_mtsp

    # indexs, cost, used_time = ortools_solve_mtsp(graph, args.agent_num, 10000)
    # env.draw(graph, cost, indexs, used_time, agent_name="or_tools")
    # print(f"or tools :{cost}")
    from model.n4Model.config import Config
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f"../log/workflow-{timestamp}")
    writer.add_text("agent_config", str(args), 0)
    agent = Agent(args, Config)
    for i in tqdm.tqdm(range(100_000_000)):
        graph = graphG.generate()
        graph_8 = graphG.augment_xy_data_by_8_fold_numpy(graph)
        agent_num = np.random.randint(args.agent_num, args.agent_num+1)
        act_logp, agents_logp, costs = agent.run_batch_episode(env, graph_8, agent_num, eval_mode=False)
        act_loss, agents_loss = agent.learn(act_logp, agents_logp, costs)
        writer.add_scalar("train/act_loss", act_loss, i)
        writer.add_scalar("train/agents_loss", agents_loss, i)
        writer.add_scalar("train/costs", np.mean(np.max(costs,axis=-1)), i)
        if (i%100) == 0:
            eval_graph = graphG.generate(1,args.city_nums)
            ortools_trajectory, ortools_cost, used_time = ortools_solve_mtsp(eval_graph, args.agent_num, 10000)
            cost,trajectory =  agent.eval_episode(env, eval_graph,agent_num, "greedy")
            print(f"agent_num:{agent_num}, act_loss:{act_loss:.5f},"
                  f" agents_loss:{agents_loss:.5f},"
                  f" costs:{cost.item():.5f},"
                  f"gap:{((cost - ortools_cost) / ortools_cost).item()*100 :.5f}%")
