
import numpy as np
import random
import torch
import sys

from sympy import floor

sys.path.append("../")
sys.path.append("./")

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
from envs.MTSP.MTSP5 import MTSPEnv
from algorithm.DNN5.AgentV8 import Agent as Agent
import tqdm
from EvalTools import EvalTools
from model.n4Model.config import Config as Config
from envs.GraphGenerator import GraphGenerator as GG


def set_seed(seed=42):
    # 基础库
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch核心设置
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU时


def worker_process(share_agent, agent_class, args, env_class, env_config, recv_pipe, queue):
    env = env_class(env_config)
    work_agent = share_agent
    # work_agent = agent_class(args)
    while True:
        graph = recv_pipe.recv()
        # work_agent.model.load_state_dict(share_agent.model.state_dict())
        features_nb, actions_nb, returns_nb, masks_nb, dones_nb = work_agent.run_batch(env, graph, args.agent_num,
                                                                              args.batch_size // args.num_worker)
        queue.put((graph, features_nb, actions_nb, returns_nb, masks_nb, dones_nb))


def eval_process(share_agent, agent_class, args, env_class, env_config, recv_model_pipe, send_result_pipe, sample_times):
    env = env_class(env_config)
    eval_agent = share_agent
    # eval_agent = agent_class(args)
    print(eval_agent.device)
    while True:
        graph = recv_model_pipe.recv()
        # eval_agent.model.load_state_dict(share_agent.model.state_dict())
        st = time.time_ns()
        greedy_cost, greedy_trajectory = eval_agent.eval_episode(env, graph, args.agent_num, exploit_mode="greedy")
        ed = time.time_ns()
        greedy_time = (ed - st) / 1e9
        min_sample_cost = np.inf
        min_sample_trajectory = None
        st = time.time_ns()
        for i in range(sample_times):
            sample_cost, sample_trajectory = eval_agent.eval_episode(env, graph, args.agent_num, exploit_mode="sample")
            if sample_cost < min_sample_cost:
                min_sample_cost = sample_cost
                min_sample_trajectory = sample_trajectory
        ed = time.time_ns()
        sample_time = (ed - st) / 1e9

        ortools_trajectory, ortools_cost, used_time = ortools_solve_mtsp(graph, args.agent_num, 10000)
        env.draw_multi(graph,
                       [ortools_cost, greedy_cost, min_sample_cost],
                       [ortools_trajectory, greedy_trajectory, min_sample_trajectory],
                       [used_time, greedy_time, sample_time],
                       ["or_tools", "greedy", "sample"]
                       )

        send_result_pipe.send(
            (greedy_cost, greedy_trajectory, min_sample_cost, min_sample_trajectory, ortools_cost, ortools_trajectory))


def train_process(share_agent, agent_class, agent_args, send_pipes, queue, eval_model_pipe, eval_result_pipe):
    # agent_args.use_gpu = False

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f"../log/workflow-{timestamp}")
    writer.add_text("agent_config", str(agent_args), 0)

    eval_count = 0
    train_count = 0
    graphG = GG(1, agent_args.city_nums)

    train_agent = agent_class(agent_args)
    model_state_dict = share_agent.model.state_dict()
    train_agent.model.load_state_dict(model_state_dict)

    for _ in tqdm.tqdm(range(100_000_000)):
        graph = graphG.generate()
        # if (train_count+1) % agent_args.num_worker == 0:
        for pipe in send_pipes:
            pipe.send(graph)

        last_state_nb_list = []
        state_nb_list = []
        actions_nb_list = []
        returns_nb_list = []
        masks_nb_list = []
        dones_nb_list = []

        for i in range(agent_args.num_worker):
            graph, features_nb, actions_nb, returns_nb, masks_nb, dones_nb = queue.get()
            last_state_nb_list.append(features_nb[0])
            state_nb_list.append(features_nb[1])
            actions_nb_list.append(actions_nb)
            returns_nb_list.append(returns_nb)
            masks_nb_list.append(masks_nb)
            dones_nb_list.append(dones_nb)
        features_nb = [np.concatenate(last_state_nb_list, axis=0), np.concatenate(state_nb_list, axis=0)]
        actions_nb = np.concatenate(actions_nb_list, axis=0)
        returns_nb = np.concatenate(returns_nb_list, axis=0)
        masks_nb = np.concatenate(masks_nb_list, axis=0)
        dones_nb = np.concatenate(dones_nb_list,axis = 0)
        train_agent.model.load_state_dict(share_agent.model.state_dict())
        loss = train_agent.learn(_convert_tensor(graph, dtype=torch.float32, device=train_agent.device, target_shape_dim=3),
                           _convert_tensor(features_nb, dtype=torch.float32, device=train_agent.device),
                           _convert_tensor(actions_nb, dtype=torch.float32, device=train_agent.device),
                           _convert_tensor(returns_nb, dtype=torch.float32, device=train_agent.device),
                           _convert_tensor(masks_nb, dtype=torch.float32, device=train_agent.device),
                            dones_nb
                           )
        writer.add_scalar("loss", loss, train_count)
        torch.cuda.empty_cache()  # 清理未使用的缓存
        share_agent.model.load_state_dict(train_agent.model.state_dict())

        if (train_count + 1) % 100 == 0:
            eval_count = train_count
            # graph = graphG.generate()
            eval_model_pipe.send(graph)

        train_count += 1

        if (train_count +1)% 100 == 0 and eval_result_pipe.poll():
            greedy_cost, greedy_trajectory, min_sample_cost, min_sample_trajectory, ortools_cost, ortools_trajectory = eval_result_pipe.recv()
            writer.add_scalar("greedy_cost", greedy_cost, eval_count)
            writer.add_scalar("min_sample_cost", min_sample_cost, eval_count)
            writer.add_scalar("ortools_cost", ortools_cost, eval_count)
            greedy_gap = (greedy_cost - ortools_cost) / ortools_cost * 100
            sample_gap = (min_sample_cost - ortools_cost) / ortools_cost * 100
            writer.add_scalar("greedy_gap", greedy_gap, eval_count)
            writer.add_scalar("sample_gap", sample_gap, eval_count)

            print(f"greddy_cost:{greedy_cost},{greedy_gap}%, sample_cost:{min_sample_cost},{sample_gap}%, ortools_cost:{ortools_cost}")

        if (train_count + 1) % 5000 == 0:
            train_agent.save_model(train_count + 1 + agent_args.agent_id)
