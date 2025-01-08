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


def worker_process(agent_class, agent_args, env_class, env_config, recv_pipe, queue):
    agent = agent_class(agent_args)
    env = env_class(env_config)
    while True:
        model_state_dict, graph = recv_pipe.recv()
        agent.model.load_state_dict(model_state_dict)
        features_nb, actions_nb, returns_nb, masks_nb = agent.run_batch(env, graph, agent_args.agent_num, agent_args.batch_size)
        queue.put((graph, features_nb, actions_nb, returns_nb, masks_nb))

def eval_process(agent_class, agent_args, env_class, env_config, recv_model_pipe, send_result_pipe, sample_times):
    agent = agent_class(agent_args)
    env = env_class(env_config)

    while True:
        # if recv_pipe.poll():
        model_state_dict, graph = recv_model_pipe.recv()
        agent.model.load_state_dict(model_state_dict)
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
    agent = agent_class(agent_args)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f"../log/workflow-{timestamp}")
    writer.add_text("agent_config", str(agent_args), 0)

    eval_count = 0
    train_count = 0
    graphG = GG(1, agent_args.city_nums)
    while True:
        for pipe in send_pipes:
            graph = graphG.generate()
            pipe.send((agent.model.state_dict(), graph))

        for _ in range(agent_args.agent_num):
            graph, features_nb, actions_nb, returns_nb, masks_nb = queue.get()
            train_count += 1
            loss = agent.learn(_convert_tensor(graph, dtype=torch.float32, device=agent.device, target_shape_dim=3),
                               _convert_tensor(features_nb, dtype=torch.float32, device=agent.device),
                               _convert_tensor(actions_nb, dtype=torch.float32, device=agent.device),
                               _convert_tensor(returns_nb, dtype=torch.float32, device=agent.device),
                               _convert_tensor(masks_nb, dtype=torch.float32, device=agent.device)
                               )
            writer.add_scalar("loss", loss, train_count)
            if train_count % 1000 == 0:
                eval_count = train_count
                graph = graphG.generate()
                eval_model_pipe.send((agent.model.state_dict(), graph))

        if eval_result_pipe.poll():
            greedy_cost, greedy_trajectory, min_sample_cost, min_sample_trajectory, ortools_cost, ortools_trajectory = eval_result_pipe.recv()
            writer.add_scalar("greedy_cost", greedy_cost, eval_count)
            writer.add_scalar("min_sample_cost", min_sample_cost, eval_count)
            writer.add_scalar("ortools_cost", ortools_cost, eval_count)



class ParallelWorker:
    def __init__(self, agent_class, agent_args, env_class, env_config, num_worker=4):
        self.agent_class = agent_class
        self.agent_args = agent_args
        self.env_class = env_class
        self.env_config = env_config
        self.num_worker = num_worker

        self.queue = mp.Queue()
        self.worker_pipes = [mp.Pipe() for _ in range(num_worker)]
        self.eval_model_pipes = mp.Pipe()
        self.eval_result_pipes = mp.Pipe()

    def run(self):
        worker_processes = [mp.Process(target=worker_process,
                                     args=(self.agent_class,
                                           self.agent_args,
                                           self.env_class,
                                           self.env_config,
                                           self.worker_pipes[worker_id][1],
                                           self.queue))
                          for worker_id in range(self.num_worker)
                          ]

        trainer_process = mp.Process(target=train_process,
                                     args=(self.agent_class,
                                           self.agent_args,
                                           [pipe[0] for pipe in self.worker_pipes],
                                           self.queue,
                                           self.eval_model_pipes[0],
                                           self.eval_result_pipes[1]
                                           )
                                     )

        evaler_process = mp.Process(target=eval_process,
                                    args=(self.agent_class,
                                          self.agent_args,
                                          self.env_class,
                                          self.env_config,
                                          self.eval_model_pipes[1],
                                          self.eval_result_pipes[0],
                                          64
                                          )
                                    )

        trainer_process.start()
        evaler_process.start()
        for p in worker_processes:
            p.start()

        trainer_process.join()
        evaler_process.join()
        for p in worker_processes:
            p.join()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_num", type=int, default=3)
    parser.add_argument("--agent_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad_max_norm", type=float, default=1.0)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--returns_norm", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=float, default=512)
    parser.add_argument("--city_nums", type=int, default=50)
    parser.add_argument("--agent_nums", type=int, default=5)
    parser.add_argument("--allow_back", type=bool, default=False)
    args = parser.parse_args()

    mp.set_start_method("spawn")

    env_config = {
        "city_nums":(args.city_nums, args.city_nums),
        "agent_nums":(args.agent_num, args.agent_num),
        "allow_back":args.allow_back,
    }
    PW = ParallelWorker(Agent, args, MTSPEnv, env_config,1)
    PW.run()