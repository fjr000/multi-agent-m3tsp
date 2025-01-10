import time

import numpy as np

import envs.MTSP.Config
from envs.MTSP.MTSP import MTSPEnv
from typing import Dict
import cloudpickle
import torch.multiprocessing as mp
from envs.GraphGenerator import GraphGenerator as GG
from utils.TensorTools import _convert_tensor
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from algorithm.OR_Tools.mtsp import ortools_solve_mtsp
import argparse
from envs.MTSP.MTSP import MTSPEnv
from algorithm.DNN.Agent_v1 import AgentV1 as Agent
import tqdm
torch.set_num_threads(1)

def worker_process(agent_class, agent_args, env_class, env_config, recv_pipe, queue):
    agent_args.use_gpu = False
    agent = agent_class(agent_args)
    env = env_class(env_config)
    model_state_dict, graph = recv_pipe.recv()
    agent.model.load_state_dict(model_state_dict)
    agent.model.to(agent.device)
    while True:
        if recv_pipe.poll():
            model_state_dict, graph = recv_pipe.recv()
            agent.model.load_state_dict(model_state_dict)
            agent.model.to(agent.device)
        features_nb, actions_nb, returns_nb, masks_nb = agent.run_batch(env, graph, agent_args.agent_num, agent_args.batch_size)
        queue.put((graph, features_nb, actions_nb, returns_nb, masks_nb))

def eval_process(agent_class, agent_args, env_class, env_config, recv_model_pipe, send_result_pipe, sample_times):
    agent_args.use_gpu = False
    agent = agent_class(agent_args)
    env = env_class(env_config)
    model_state_dict, graph = recv_model_pipe.recv()
    agent.model.load_state_dict(model_state_dict)
    agent.model.to(agent.device)
    while True:
        model_state_dict, graph = recv_model_pipe.recv()
        agent.model.load_state_dict(model_state_dict)
        agent.model.to(agent.device)

        greedy_cost, greedy_trajectory = agent.eval_episode(env, graph, agent_args.agent_num, exploit_mode="greedy")
        min_sample_cost = np.inf
        min_sample_trajectory = None
        for i in range(sample_times):
            sample_cost, sample_trajectory = agent.eval_episode(env, graph, agent_args.agent_num, exploit_mode="sample")
            if sample_cost < min_sample_cost:
                min_sample_cost = sample_cost
                min_sample_trajectory = sample_trajectory

        ortools_trajectory, ortools_cost, used_time = ortools_solve_mtsp(graph, agent_args.agent_num, 10000)
        # env.draw(graph, cost, indexs, used_time, agent_name="or_tools")

        send_result_pipe.send((greedy_cost, greedy_trajectory, min_sample_cost, min_sample_trajectory, ortools_cost, ortools_trajectory))



def train_process(agent_class, agent_args, send_pipes, queue, eval_model_pipe, eval_result_pipe):
    # agent_args.use_gpu = False

    agent = agent_class(agent_args)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f"../log/workflow-{timestamp}")
    writer.add_text("agent_config", str(agent_args), 0)

    eval_count = 0
    train_count = 0
    graphG = GG(1, agent_args.city_nums)

    model_state_dict = agent.model.state_dict()
    model_state_dict_cpu = {k: v.cpu() for k, v in model_state_dict.items()}
    graph = graphG.generate()

    for pipe in send_pipes:
        pipe.send((model_state_dict_cpu, graph))

    for _ in tqdm.tqdm(range(100_000_000)):

        if (train_count+1) % agent_args.num_worker == 0:
            model_state_dict = agent.model.state_dict()
            model_state_dict_cpu = {k: v.cpu() for k, v in model_state_dict.items()}
            for pipe in send_pipes:
                pipe.send((model_state_dict_cpu, graph))

        graph, features_nb, actions_nb, returns_nb, masks_nb = queue.get()
        loss = agent.learn(_convert_tensor(graph, dtype=torch.float32, device=agent.device, target_shape_dim=3),
                           _convert_tensor(features_nb, dtype=torch.float32, device=agent.device),
                           _convert_tensor(actions_nb, dtype=torch.float32, device=agent.device),
                           _convert_tensor(returns_nb, dtype=torch.float32, device=agent.device),
                           _convert_tensor(masks_nb, dtype=torch.float32, device=agent.device)
                           )
        writer.add_scalar("loss", loss, train_count)

        if (train_count+1) % 100 == 0:
            eval_count = train_count
            # graph = graphG.generate()
            model_state_dict = agent.model.state_dict()
            model_state_dict_cpu = {k: v.cpu() for k, v in model_state_dict.items()}
            eval_model_pipe.send((model_state_dict_cpu, graph))

        train_count += 1

        if eval_result_pipe.poll():
            greedy_cost, greedy_trajectory, min_sample_cost, min_sample_trajectory, ortools_cost, ortools_trajectory = eval_result_pipe.recv()
            writer.add_scalar("greedy_cost", greedy_cost, eval_count)
            writer.add_scalar("min_sample_cost", min_sample_cost, eval_count)
            writer.add_scalar("ortools_cost", ortools_cost, eval_count)
            print(f"greddy_cost:{greedy_cost}, sample_cost:{min_sample_cost}, ortools_cost:{ortools_cost}")



class ParallelWorker:
    def __init__(self, agent_class, agent_args, env_class, env_config):
        self.agent_class = agent_class
        self.agent_args = agent_args
        self.env_class = env_class
        self.env_config = env_config
        self.num_worker = agent_args.num_worker

        self.queue = mp.Queue()
        self.worker_pipes = [mp.Pipe(duplex=False) for _ in range(self.num_worker)]
        self.eval_model_pipes = mp.Pipe(duplex=False)
        self.eval_result_pipes = mp.Pipe(duplex=False)

    def run(self):
        worker_processes = [mp.Process(target=worker_process,
                                     args=(self.agent_class,
                                           self.agent_args,
                                           self.env_class,
                                           self.env_config,
                                           self.worker_pipes[worker_id][0],
                                           self.queue))
                          for worker_id in range(self.num_worker)
                          ]

        trainer_process = mp.Process(target=train_process,
                                     args=(self.agent_class,
                                           self.agent_args,
                                           [pipe[1] for pipe in self.worker_pipes],
                                           self.queue,
                                           self.eval_model_pipes[1],
                                           self.eval_result_pipes[0]
                                           )
                                     )

        evaler_process = mp.Process(target=eval_process,
                                    args=(self.agent_class,
                                          self.agent_args,
                                          self.env_class,
                                          self.env_config,
                                          self.eval_model_pipes[0],
                                          self.eval_result_pipes[1],
                                          64
                                          )
                                    )

        trainer_process.start()
        # time.sleep(10)
        for p in worker_processes:
            p.start()
        # time.sleep(10)
        evaler_process.start()

        trainer_process.join()
        for p in worker_processes:
            p.join()

        evaler_process.join()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_worker", type=int, default=1)
    parser.add_argument("--agent_num", type=int, default=5)
    parser.add_argument("--agent_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad_max_norm", type=float, default=1.0)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--returns_norm", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=float, default=256)
    parser.add_argument("--city_nums", type=int, default=50)
    parser.add_argument("--allow_back", type=bool, default=False)
    args = parser.parse_args()

    mp.set_start_method("spawn")

    env_config = {
        "city_nums":(args.city_nums, args.city_nums),
        "agent_nums":(args.agent_num, args.agent_num),
        "allow_back":args.allow_back,
    }
    PW = ParallelWorker(Agent, args, MTSPEnv, env_config)
    PW.run()