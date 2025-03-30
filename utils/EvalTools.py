import os

import numpy as np

from utils.TspInstanceFileTool import TspInstanceFileTool
import time
from algorithm.OR_Tools.mtsp import ortools_solve_mtsp
from utils.TspInstanceFileTool import TspInstanceFileTool, result_dict
from envs.GraphGenerator import GraphGenerator as GG

class EvalTools(object):

    @staticmethod
    def CheckGraph(graph):
        B = 0
        if len(graph.shape) == 2:
            B = 1
            graph = graph[np.newaxis,]
        elif len(graph.shape) == 3:
            B = graph.shape[0]
        else:
            raise Exception('Graph must be 2 or 3 dimensional')
        return B, graph
    @staticmethod
    def EvalLKH3(graph, agent_num, instance_name = "mtsp"):
        B, graph = EvalTools.CheckGraph(graph)

        LKH_cost_list = []
        LKH_traj_list = []
        LKH_traj = None
        st = time.time_ns()

        par = instance_name + ".par"
        tsp = instance_name + ".tsp"
        tour = instance_name + ".tour"

        for k in range(B):
            os.system("rm " + par + " > /dev/null")
            os.system("rm " + tsp + " > /dev/null")
            os.system("rm " + tour + " > /dev/null")
            TspInstanceFileTool.writeTSPInstanceToFile(tsp, graph[k], (1,), instance_name)
            TspInstanceFileTool.writeLKH3MTSPPar(par, tsp, agent_num, 1, output_filename=tour)
            pwd = "./LKH "
            os.system(pwd + par + " > /dev/null")
            LKH_min_cost, _, LKH_traj = TspInstanceFileTool.readLKH3Route(tour)
            LKH_cost_list.append(LKH_min_cost)
            LKH_traj_list.append(LKH_traj)
        ed = time.time_ns()
        LKH_min_cost = np.mean(LKH_cost_list)
        LKH3_time = (ed - st) / 1e9

        if B == 1:
            return LKH_min_cost, LKH_traj, LKH3_time
        else:
            return LKH_min_cost, LKH_traj_list, LKH3_time

    @staticmethod
    def EvalOrTools(graph, agent_num, instance_name = "mtsp"):
        B, graph = EvalTools.CheckGraph(graph)

        ortool_cost_list=[]
        ortool_traj_list=[]
        ortools_trajectory = None
        st = time.time_ns()
        for k in range(B):
            ortools_trajectory, ortools_cost, _ = ortools_solve_mtsp(graph[k:k+1], agent_num, 10000)
            ortool_cost_list.append(ortools_cost)
            ortool_traj_list.append(ortools_trajectory)

        ortools_cost_mean = np.mean(ortool_cost_list)
        ed = time.time_ns()
        ortools_time = (ed - st) / 1e9
        if B == 1:
            return ortools_cost_mean, ortools_trajectory, ortools_time
        else:
            return ortools_cost_mean, ortool_traj_list, ortools_time


    @staticmethod
    def EvalGreedy(graph, agent_num, agent, env, info = None, instance_name = "mtsp", aug = True):
        B, graph = EvalTools.CheckGraph(graph)

        st = time.time_ns()
        if aug:
            graph = GG.augment_xy_data_by_8_fold_numpy(graph)
        cost,greedy_trajectory = agent.eval_episode(env, graph, agent_num, "greedy", info = info)
        ed = time.time_ns()
        if aug:
            cost_8 = cost.reshape(B,8,-1)
            min_max_arg = np.argmin(cost_8, axis = 1)
            cost = cost_8[np.arange(B)[:, None],min_max_arg]
            greedy_trajectory_8 = greedy_trajectory.reshape(B,8,greedy_trajectory.shape[1],greedy_trajectory.shape[2])
            greedy_trajectory = greedy_trajectory_8[np.arange(B)[:, None],min_max_arg].squeeze(1)

        greedy_cost = np.mean(cost)
        # greedy_cost = cost.squeeze()
        greedy_time = (ed- st) / 1e9
        traj = env.compress_adjacent_duplicates_optimized(greedy_trajectory)
        if B == 1:
            return greedy_cost, traj[0], greedy_time
        else:
            return greedy_cost, traj, greedy_time

    @staticmethod
    def EvalSample(graph, agent_num, agent, env, info = None, run_times = 32,instance_name = "mtsp"):
        B, graph = EvalTools.CheckGraph(graph)

        st = time.time_ns()
        batch_eval_graph = graph[np.newaxis,].repeat(run_times, axis=0)
        batch_eval_graph = batch_eval_graph.reshape(run_times * graph.shape[0],graph.shape[1],graph.shape[2])
        cost, trajectory = agent.eval_episode(env, batch_eval_graph,agent_num, "sample")
        batch_cost = cost.reshape(run_times, graph.shape[0], -1,)
        min_sample_cost = np.min(batch_cost, axis=0)
        min_sample_cost_mean = np.mean(min_sample_cost)
        ed = time.time_ns()
        sample_time = (ed - st) / 1e9

        if B == 1:
            idx = np.argmin(batch_cost, axis=0).item()
            min_sample_traj = trajectory[idx:idx + 1]
            sample_traj = env.compress_adjacent_duplicates_optimized(min_sample_traj)[0]

            return min_sample_cost_mean, sample_traj, sample_time
        else:
            return min_sample_cost_mean, None, sample_time

    @staticmethod
    def tensorboard_write(writer, train_count, act_loss, agents_loss, act_ent_loss, agents_ent_loss, lr):
        writer.add_scalar("train/act_loss", act_loss, train_count)
        writer.add_scalar("train/agents_loss", agents_loss, train_count)
        writer.add_scalar("train/act_ent_loss", act_ent_loss, train_count)
        writer.add_scalar("train/agt_ent_loss", agents_ent_loss, train_count)
        writer.add_scalar("train/lr", lr, train_count)

    @staticmethod
    def eval_mtsplib(agent, env, writer, step):
        for graph_name in ("eil51", "berlin52", "eil76", "rat99"):
            graph, scale = TspInstanceFileTool.loadTSPLib("../graph/tsp", graph_name)
            for agent_num in (2, 3, 5, 7):
                greedy_cost, greedy_traj, greedy_time = EvalTools.EvalGreedy(graph, agent_num, agent, env)
                # no_conflict_greedy_cost, no_conflict_greedy_traj, no_conflict_greedy_time = EvalTools.EvalGreedy(graph, agent_num, agent, env, {"use_conflict_model": False})
                best_cost = result_dict[graph_name][agent_num][0] / scale
                writer.add_scalar(f"eval/{graph_name}/{agent_num}/greedy_gap",
                                  (greedy_cost - best_cost) / best_cost * 100, step)
                # writer.add_scalar(f"eval/{graph_name}/{agent_num}/no_conflict_cost", (no_conflict_greedy_cost - best_cost) / best_cost * 100, step)

    @staticmethod
    def eval_random(graph, agent_num, agent, env, writer, step,draw = False):
        greedy_cost, greedy_traj, greedy_time = EvalTools.EvalGreedy(graph, agent_num, agent, env)
        if agent_num == 1:
            cost, traj, time = EvalTools.EvalOrTools(graph, agent_num)
        else:
            cost, traj, time = EvalTools.EvalLKH3(graph, agent_num)

        greedy_gap = ((greedy_cost - cost) / cost).item() * 100
        writer.add_scalar("eval/gap", greedy_gap, step)
        print(f"agent_num:{agent_num},city_num:{graph.shape[1]},"
              f"greedy_gap:{greedy_gap:.5f}%,"
              f"costs:{greedy_cost.item():.5f},"
              f"LKH3_OrTools_costs:{cost:.5f}"
              )
        print(f"traj:\n{greedy_traj}")
        if draw:
            env.draw_multi(
                graph,
                [greedy_cost, cost],
                [greedy_traj, traj],
                [greedy_time, time],
                ["greedy", "or_tools" if agent_num == 1 else "LKH3"]
            )

        return greedy_gap

    @staticmethod
    def EvalTSPGreedy(graph, agent, traj, info = None, instance_name = "tsp", aug = True):
        from envs.TSP.TSP import TSPEnv
        tsp = TSPEnv({})
        if aug:
            expanded = []
            for b in traj:  # 遍历每个 B
                for _ in range(8):  # 重复 8 次
                    expanded.append(b)  # 直接追加（浅拷贝）
            traj = expanded
        tsp_cost, tsp_traj, tsp_time = EvalTools.EvalGreedy(graph, 1, agent, tsp, info={"trajs": traj}, aug=aug)
        return tsp_cost, tsp_traj, tsp_time
