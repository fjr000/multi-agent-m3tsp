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
from utils.EvalTools import EvalTools


def tensorboard_write(writer, train_count, act_loss, agents_loss, act_ent_loss, agents_ent_loss, costs, lr):
    writer.add_scalar("train/act_loss", act_loss, train_count)
    writer.add_scalar("train/agents_loss", agents_loss, train_count)
    writer.add_scalar("train/act_ent_loss", act_ent_loss, train_count)
    writer.add_scalar("train/agt_ent_loss", agents_ent_loss, train_count)
    writer.add_scalar("train/costs", np.mean(np.max(costs, axis=-1)), train_count)
    writer.add_scalar("train/lr", lr, train_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--agent_num", type=int, default=10)
    parser.add_argument("--fixed_agent_num", type=bool, default=False)
    parser.add_argument("--agent_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_max_norm", type=float, default=10)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=1e-2)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--city_nums", type=int, default=40)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=80000)
    parser.add_argument("--env_masks_mode", type=int, default=1,
                        help="0 for only the min cost  not allow back depot; 1 for only the max cost allow back depot")
    parser.add_argument("--eval_interval", type=int, default=100, help="eval  interval")
    parser.add_argument("--use_conflict_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--only_one_instance", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--save_model_interval", type=int, default=10000, help="save model interval")
    args = parser.parse_args()

    from envs.GraphGenerator import GraphGenerator as GG

    fig = None
    graphG = GG(args.batch_size, args.city_nums, 2)
    from envs.MTSP.MTSP4 import MTSPEnv

    env = MTSPEnv({"env_masks_mode": args.env_masks_mode})
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
    for lr in agent.optim.param_groups:
        lr["lr"] = args.lr
    from CourseController import CourseController

    CC = CourseController()
    agent_num, city_nums = args.agent_num, args.city_nums
    for i in tqdm.tqdm(range(100_000_000), mininterval=1):
        # agent_num, city_nums = CC.get_course()
        if args.only_one_instance:
            graph = graphG.generate(1).repeat(args.batch_size, axis=0)
            graph_8 = graphG.augment_xy_data_by_8_fold_numpy(graph)
        else:
            graph = graphG.generate(args.batch_size, city_nums)
            graph_8 = graphG.augment_xy_data_by_8_fold_numpy(graph)

        if args.fixed_agent_num:
            agent_num = np.random.randint(args.agent_num, args.agent_num + 1)
        else:
            agent_num = np.random.randint(2, args.agent_num + 1)

        act_logp, agents_logp, act_ent, agt_ent, costs = agent.run_batch_episode(env, graph_8, agent_num,
                                                                                 eval_mode=False, info={
                "use_conflict_model": args.use_conflict_model})
        act_loss, agents_loss, act_ent_loss, agt_ent_loss = agent.learn(act_logp, agents_logp, act_ent, agt_ent, costs)

        tensorboard_write(writer, i,
                          act_loss, agents_loss,
                          act_ent_loss, agt_ent_loss,
                          costs, agent.optim.param_groups[0]["lr"]
                          )

        if ((i + 1) % args.eval_interval) == 0:
            eval_graph = graphG.generate(1, city_nums)
            LKH3_cost, LKH3_traj, LKH3_time = EvalTools.EvalLKH3(eval_graph, agent_num)
            greedy_cost,  greedy_traj, greedy_time = EvalTools.EvalGreedy(eval_graph, agent_num, agent, env)
            no_conflict_cost,  no_conflict_trajectory, no_conflict_time = EvalTools.EvalGreedy(eval_graph, agent_num, agent, env, {"use_conflict_model": False})

            greedy_gap = ((greedy_cost - LKH3_cost) / LKH3_cost).item() * 100
            no_conflict_gap = ((no_conflict_cost - LKH3_cost) / LKH3_cost).item() * 100
            # fig = env.draw(eval_graph[0],cost.item(), traj[0],gap)
            # if gap < 0 or no_conflict_gap < 0:
            #     env.draw_multi(
            #         eval_graph[0],
            #         [cost.item(), no_conflict_cost.item(), LKH3_cost],
            #         [traj[0], no_conflict_traj[0], LKH3_traj],
            #         [0, 0, 0],
            #         ["greedy", "no_conflict_model", 'LKH3']
            #     )
            CC.update_result(greedy_gap / 100)
            writer.add_scalar("eval/gap", greedy_gap, i)
            writer.add_scalar("eval/no_conflict_gap", no_conflict_gap, i)
            agent.lr_scheduler.step(greedy_gap)
            print(f"agent_num:{agent_num},city_num:{city_nums}, "
                  f"greedy_gap:{greedy_gap:.5f}%, no_conflict_gap:{no_conflict_gap:.5f}%, "
                  f"costs:{greedy_cost.item():.5f}, no_conflict_costs:{no_conflict_cost.item():.5f},"
                  f"LKH3_costs:{LKH3_cost:.5f}"
                  )

        if (i + 1) % (args.save_model_interval * args.accumulation_steps) == 0:
            agent.save_model(args.agent_id + i + 1)
