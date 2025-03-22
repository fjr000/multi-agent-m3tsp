import argparse
from algorithm.DNN5.AgentV1 import AgentV1 as Agent
from model.n4Model.config import Config as Config

from utils.EvalTools import EvalTools
from utils.TspInstanceFileTool import TspInstanceFileTool, result_dict
if __name__ == '__main__':
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
    parser.add_argument("--grad_max_norm", type=float, default=1)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=1e-2)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--city_nums", type=int, default=50)
    parser.add_argument("--random_city_num", type=bool, default=True)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=80000)
    parser.add_argument("--env_masks_mode", type=int, default=2,
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


    from envs.MTSP.MTSP5 import MTSPEnv

    env = MTSPEnv({"env_masks_mode":args.env_masks_mode})
    agent = Agent(args, Config)
    agent.load_model(args.agent_id)

    eval_results = {}

    for graph_name in ("eil51", "berlin52", "eil76", "rat99"):
        graph, scale = TspInstanceFileTool.loadTSPLib("../graph/tsp",graph_name)
        for agent_num in (2,3,5,7):
            greedy_cost, greedy_traj, greedy_time = EvalTools.EvalGreedy(graph, agent_num, agent, env)
            no_conflict_greedy_cost, no_conflict_greedy_traj, no_conflict_greedy_time = EvalTools.EvalGreedy(graph, agent_num, agent, env, {"use_conflict_model": False})
            LKH_cost, LKH_traj, LKH_time = EvalTools.EvalLKH3(graph, agent_num)
            ortools_cost, ortools_traj, ortools_time = EvalTools.EvalOrTools(graph, agent_num)
            sample_cost, sample_traj, sample_time = EvalTools.EvalSample(graph, agent_num, agent, env)

            if args.draw:
                env.draw_multi(
                    graph,
                    [greedy_cost, no_conflict_greedy_cost, sample_cost, ortools_cost, LKH_cost],
                    [greedy_traj, no_conflict_greedy_traj, sample_traj, ortools_traj, LKH_traj],
                    [greedy_time, no_conflict_greedy_time, sample_time, ortools_time, LKH_time],
                    ["greedy","no_conflict_greedy", "sample","or_tools", "LKH"]
                )
            best_cost = result_dict[graph_name][agent_num][0] / scale
            print(f"graph:{graph_name}, agent_num:{agent_num},"
                  f"greedy_gap:{(greedy_cost-best_cost) / best_cost * 100 :.5f} %,"
                  f"no_conflict_greedy_gap:{(no_conflict_greedy_cost -best_cost) / best_cost * 100 :.5f} %,"
                  f"sample_gap:{(sample_cost -best_cost) / best_cost * 100 :.5f} %,"
                  f"ortools_gap:{(ortools_cost -best_cost) / best_cost *100 :.5f} %,"
                  f"LKH_gap:{(LKH_cost -best_cost) / best_cost *100 :.5f} %")





