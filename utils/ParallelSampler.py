import time
from random import random

import envs.MTSP.Config
from envs.MTSP.MTSP import MTSPEnv
from typing import Dict
import torch.multiprocessing as multiprocessing
import numpy as np
import copy
import cloudpickle

class ParallelSampler:
    def __init__(self, agent, env_class=MTSPEnv, num_worker=2, config: Dict = None):
        self.agent = agent
        self.agent_id = 0
        # self.lock = multiprocessing.Lock()
        self.env_class = env_class
        self.num_worker = num_worker
        self.config = config
        self.city_nums = None
        self.agents_num = None
        self.seed = None
        self.processes = [None for _ in range(num_worker)]
        self.queue = multiprocessing.Queue()
        self.agent_pipes = [multiprocessing.Pipe() for _ in range(num_worker)]
        self.envs = []
        # np.random.seed(config["seed"])
        for i in range(self.num_worker):
            cfg = config
            cfg["seed"] = config["seed"] + i
            # cfg["seed"] = None
            self.envs.append(self.env_class(cfg))

    def update_agent(self, id, agent):
        for i in range(self.num_worker):
            data = cloudpickle.dumps((id, agent))
            self.agent_pipes[i][0].send(data)

    def __update_agent(self, worker_id):
        while self.agent_pipes[worker_id][1].poll():
            data = self.agent_pipes[worker_id][1].recv()
            self.agent_id, self.agent = cloudpickle.loads(data)

    def _run_episode(self, worker_id):

        self.__update_agent(worker_id)
        # print(f"{worker_id} 's agent id: {self.agent_id}")
        env = self.envs[worker_id]
        obs, info = env.reset()
        anum = info.get("anum")
        cnum = info.get("cnum")
        graph = info.get("graph")
        graph_matrix = info.get("graph_matrix")
        global_mask = info.get("global_mask")
        agents_action_mask = info.get("agents_action_mask")
        agent_config = {
            "city_nums": cnum,
            "agent_nums": anum
        }

        self.agent.reset(agent_config)

        done = False
        obs_list = [obs]
        reward_list = []
        done_list = []
        global_mask_list = []
        action_mask_list = []
        global_info = {
            "graph": graph,
            "graph_matrix": graph_matrix,
        }
        while not done:
            act = self.agent.predict(obs, global_mask, agents_action_mask)
            next_obs, reward, done, info = env.step(act)
            obs_list.append(next_obs)
            reward_list.append(reward)
            done_list.append(done)
            global_mask = info.get("global_mask")
            agents_action_mask = info.get("agents_action_mask")
            global_mask_list.append(global_mask)
            action_mask_list.append(agents_action_mask)
            obs = next_obs
        global_info["actors_cost"] = info["actors_cost"]
        global_info["actors_trajectory"] = info["actors_trajectory"]
        global_info["actors_action_record"] = info["actors_action_record"]
        self.queue.put((worker_id, obs_list, reward_list, done_list, global_mask_list, action_mask_list, global_info))

    def start(self):
        for i in range(self.num_worker):
            p = multiprocessing.Process(target=self._run_episode, args=(i,))
            self.processes[i] = p
            p.start()

    def _sample(self):
        obs_lists = []
        reward_lists = []
        done_lists = []
        global_mask_lists = []
        action_mask_lists = []
        global_info_list = []
        try:
            for _ in range(self.num_worker):
                qlen = self.queue.qsize()
                worker_id, obs_list, reward_list, done_list, global_mask_list, action_mask_list, global_info = self.queue.get()
                obs_lists.append(obs_list)
                reward_lists.append(reward_list)
                done_lists.append(done_list)
                global_mask_lists.append(global_mask_list)
                action_mask_lists.append(action_mask_list)
                global_info_list.append(global_info)
                # print(f"before_size:{qlen}, after_size:{self.queue.qsize()}")
        except KeyboardInterrupt:
            print("采样中断.")
        return obs_lists, reward_lists, done_lists, global_mask_lists, action_mask_lists, global_info_list

    def collect(self):
        res = self._sample()

        self.close()

        return res

    def close(self):
        for p in self.processes:
            if p is not None:
                p.terminate()
                p.join()


class ParallelSamplerAsync(ParallelSampler):
    def __init__(self, agent, env_class=MTSPEnv, num_worker=2, config: Dict = None):
        super(ParallelSamplerAsync, self).__init__(agent, env_class, num_worker, config)
        self.is_running = False

    def __run(self, worker_id):
        while True:
            if self.queue.qsize() > self.num_worker * 2:
                time.sleep(1)
            else:
                self._run_episode(worker_id)

    def start(self):
        if self.is_running:
            return

        for i in range(self.num_worker):
            p = multiprocessing.Process(target=self.__run, args=(i,))
            self.processes[i] = p
            p.start()
        self.is_running = True

    def collect(self):
        return self._sample()


if __name__ == '__main__':
    from model.RandomAgent import RandomAgent
    from utils.GraphPlot import GraphPlot as GP
    agent = RandomAgent()
    num_worker = 16
    sample_times = 100
    gp = GP()
    PS = ParallelSamplerAsync(agent, MTSPEnv, num_worker=num_worker, config=envs.MTSP.Config.Config)
    st=time.time_ns()
    for t in range(sample_times):
        PS.start()
        obs_lists, reward_lists, done_lists, global_mask_lists, action_mask_lists, global_info_list = PS.collect()
        # for i in range(num_worker):
        #     global_info = global_info_list[i]
        #     gp.draw_route(global_info["graph"], global_info["actors_trajectory"], one_first=True)
        PS.update_agent(t + 1,agent)
    ed=time.time_ns()
    PS.close()
    # for i in range(num_worker):
    #     global_info = global_info_list[i]
    #     gp.draw_route(global_info["graph"], global_info["actors_trajectory"], one_first=True)
    print(f"time_cost_per_worker:{(ed-st)/1e9 / num_worker / sample_times}")
