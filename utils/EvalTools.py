import os

import numpy as np

from utils.TspInstanceFileTool import TspInstanceFileTool
import time
from algorithm.OR_Tools.mtsp import ortools_solve_mtsp

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
    def EvalGreedy(graph, agent_num, agent, env, info = None, instance_name = "mtsp"):
        B, graph = EvalTools.CheckGraph(graph)

        st = time.time_ns()
        cost,greedy_trajectory = agent.eval_episode(env, graph, agent_num, "greedy", info = info)
        ed = time.time_ns()
        greedy_cost = np.mean(cost)
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
