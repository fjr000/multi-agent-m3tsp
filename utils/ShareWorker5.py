import sys
import time

import matplotlib.pyplot as plt
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
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad_max_norm", type=float, default=1.0)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--returns_norm", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=5e-3)
    parser.add_argument("--batch_size", type=float, default=16)
    parser.add_argument("--city_nums", type=int, default=50)
    parser.add_argument("--allow_back", type=bool, default=False)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=0)
    args = parser.parse_args()

    from envs.GraphGenerator import GraphGenerator as GG
    fig = None
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
    agent.load_model(args.agent_id)
    from CourseController import CourseController
    CC = CourseController()
    agent_num, city_nums = args.agent_num, args.city_nums
    for i in tqdm.tqdm(range(100_000_000), mininterval=10):
        # agent_num, city_nums = CC.get_course()
        graph = graphG.generate(args.batch_size, city_nums)
        graph_8 = graphG.augment_xy_data_by_8_fold_numpy(graph)
        # agent_num = np.random.randint(1, args.agent_num+1)

        act_logp, agents_logp, act_ent, agt_ent, costs = agent.run_batch_episode(env, graph_8, agent_num, eval_mode=False)
        act_loss, agents_loss, act_ent_loss, agt_ent_loss = agent.learn(act_logp, agents_logp, act_ent, agt_ent, costs)
        writer.add_scalar("train/act_loss", act_loss, i)
        writer.add_scalar("train/agents_loss", agents_loss, i)
        writer.add_scalar("train/act_ent_loss", act_ent_loss, i)
        writer.add_scalar("train/agt_ent_loss", agt_ent_loss, i)
        writer.add_scalar("train/costs", np.mean(np.max(costs,axis=-1)), i)
        if (i%100) == 0:
            plt.close()
            eval_graph = graphG.generate(1,city_nums)
            ortools_trajectory, ortools_cost, used_time = ortools_solve_mtsp(eval_graph, agent_num, 10000)
            cost,trajectory =  agent.eval_episode(env, eval_graph,agent_num, "greedy")
            traj = env.compress_adjacent_duplicates_optimized(trajectory)
            gap = ((cost - ortools_cost) / ortools_cost).item()*100
            # fig = env.draw(eval_graph[0],cost.item(), traj[0],gap)
            CC.update_result(gap / 100)
            writer.add_scalar("eval/gap", gap, i)

            print(f"agent_num:{agent_num},city_num:{city_nums} "
                  f"act_loss:{act_loss:.5f},"
                  f" agents_loss:{agents_loss:.5f},"
                  f"act_ent_loss:{act_ent_loss},"
                  f"agt_ent_loss:{agt_ent_loss},"
                  f" costs:{cost.item():.5f},"
                  f"gap:{ gap:.5f}%")

        if (i+1)%10000 == 0:
            agent.save_model(i+1)
