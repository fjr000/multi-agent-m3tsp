import time

import envs.MTSP.Config
from envs.MTSP.MTSP import MTSPEnv
from typing import Dict
import cloudpickle
import torch.multiprocessing as mp
from envs.GraphGenerator import GraphGenerator as GG
from utils.TensorTools import _convert_tensor
import torch

def worker_process(agent_class, agent_args, env_class, env_config, recv_pipe, queue):
    agent = agent_class(agent_args)
    env = env_class(env_config)
    while True:
        model_state_dict, graph = recv_pipe.recv()
        agent.load_state_dict(model_state_dict)
        features_nb, actions_nb, returns_nb, masks_nb = agent.run_batch(env, graph, agent_args.agent_num)
        queue.put((graph, features_nb, actions_nb, returns_nb, masks_nb))


def train_process(agent_class, agent_args, send_pipe, queue, num_worker):
    agent = agent_class(agent_args)
    while True:
        for i in range(num_worker):
            graphG = GG(1, 50, 5)
            graph = graphG.generate()
            send_pipe.send((agent.state_dict(), graph))

        while not queue.empty():
            features_nb, actions_nb, returns_nb, masks_nb = queue.get()
            loss = agent.learn(_convert_tensor(graph, dtype=torch.float32, device=agent.device, target_shape_dim=3),
                               _convert_tensor(features_nb, dtype=torch.float32, device=agent.device),
                               _convert_tensor(actions_nb, dtype=torch.float32, device=agent.device),
                               _convert_tensor(returns_nb, dtype=torch.float32, device=agent.device),
                               _convert_tensor(masks_nb, dtype=torch.float32, device=agent.device)
                               )


class ParallelWorker:
    def __init__(self, agent_class, agent_args, env_class, env_config, num_worker=4):
        self.agent_class = agent_class
        self.agent_args = agent_args
        self.env_class = env_class
        self.env_config = env_config
        self.num_worker = num_worker

        self.queue = mp.Queue()
        self.agent_pipes = [mp.Pipe() for _ in range(num_worker)]
        self.processes = None

    def run(self):
        self.processes = [mp.Process(target=worker_process,
                                     args=(self.agent_class,
                                           self.agent_args,
                                           self.env_class,
                                           self.env_config,
                                           self.agent_pipes[worker_id][1],
                                           self.queue))
                          for worker_id in range(self.num_worker)
                          ]
        for p in self.processes:
            p.start()

    def _worker_sample(self, worker_id):
        model_state = self.agent_pipes[worker_id].recv()
        self.agents[worker_id].load_state_dict(model_state)
