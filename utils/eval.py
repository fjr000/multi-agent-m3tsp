import sys

from click.core import batch

sys.path.append("../")
sys.path.append("./")

import argparse
from envs.MTSP.MTSP5 import MTSPEnv

from algorithm.DNN5.AgentV6 import Agent as Agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--agent_num", type=int, default=5)
    parser.add_argument("--fixed_agent_num", type=bool, default=False)
    parser.add_argument("--agent_dim", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--grad_max_norm", type=float, default=0.5)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=1e-3)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--eval_size", type=int, default=200)
    parser.add_argument("--city_nums", type=int, default=50)
    parser.add_argument("--random_city_num", type=bool, default=True)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=2195004) # 710 2.493 / 850 2.489 910 2.486
    parser.add_argument("--tsp_agent_id", type=int, default=00)
    parser.add_argument("--env_masks_mode", type=int, default=7,
                        help="0 for only the min cost  not  allow back depot; 1 for only the max cost allow back depot")
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
    parser.add_argument("--save_model_interval", type=int, default=13000, help="save model interval")
    parser.add_argument("--seed", type=int, default=3333, help="random seed")
    parser.add_argument("--draw", type=bool, default=True, help="whether to draw result")

    args = parser.parse_args()

    from envs.GraphGenerator import GraphGenerator as GG
    fig = None
    graphG = GG(args.batch_size, args.city_nums, 2)

    env = MTSPEnv({"env_masks_mode":args.env_masks_mode})
    from algorithm.OR_Tools.mtsp import ortools_solve_mtsp
    from model.n4Model.config import Config
    agent = Agent(args, Config)
    agent.load_model(args.agent_id)

    if args.tsp_agent_id > 0:
        from algorithm.DNN5.AgentTSP import Agent as AgentTSP
        TSP_agent = AgentTSP(args, Config)
        TSP_agent.load_model(args.tsp_agent_id)

    city_nums = args.city_nums
    agent_nums = args.agent_num
    from EvalTools import EvalTools
    import numpy as np
    np.random.seed(args.seed)
    for i in (range(100_000_000)):
        graph = graphG.generate()

        costs = []
        trajs = []
        times = []
        names = []

        eval_graph = graph

        no_conflict_greedy_cost, no_conflict_greedy_traj, no_conflict_greedy_time=EvalTools.EvalGreedy(eval_graph, agent_nums, agent, env,{"use_conflict_model":False})
        costs.append(no_conflict_greedy_cost)
        trajs.append(no_conflict_greedy_traj[0])
        times.append(no_conflict_greedy_time)
        names.append("no_conflict_greedy")

        greedy_cost, greedy_traj, greedy_time = EvalTools.EvalGreedy(eval_graph, agent_nums, agent, env)
        costs.append(greedy_cost)
        trajs.append(greedy_traj[0])
        times.append(greedy_time)
        names.append("greedy")


        #
        # if args.tsp_agent_id > 0:
        #     optim_cost, optim_traj, optim_time = EvalTools.EvalTSPGreedy(eval_graph, TSP_agent, [greedy_traj[0]] if args.batch_size == 1 else greedy_traj)
        #     optim_time = greedy_time + optim_time
        #     costs.append(optim_cost)
        #     trajs.append(optim_traj[0])
        #     times.append(optim_time)
        #     names.append("optim")

        # sample_cost, sample_traj ,sample_time = EvalTools.EvalSample(graph, agent_nums, agent, env)
        # costs.append(sample_cost)
        # trajs.append(sample_traj)
        # times.append(sample_time)
        # names.append("sample")
        #
        # ortools_cost, ortools_traj, ortools_time = EvalTools.EvalOrTools(graph, agent_nums)
        # LKH_cost, LKH_traj, LKH_time = EvalTools.EvalLKH3(eval_graph, agent_nums)
        # costs.append(LKH_cost)
        # trajs.append(LKH_traj)
        # times.append(LKH_time)
        # names.append("lkh")

        print(costs)
        print(times)
        print(names)
        if args.batch_size == 1:
            env.draw_multi(
                graph,
                costs,# ortools_cost],
                trajs,# ortools_traj],
                times,# ortools_time],
                names,#"or_tools"],
            )
            from utils.anima import visualize_agent_trajectories
            ani = visualize_agent_trajectories(graph[0], greedy_traj[1][0],f"anime{i}.gif")