import math

import numpy as np
from typing import Tuple, List, Dict
import sys
import gym

sys.path.append("../")
from envs.MTSP.Config import Config
from envs.GraphGenerator import GraphGenerator as GG
from utils.GraphPlot import GraphPlot as GP
from model.NNN.RandomAgent import RandomAgent
import torch.nn.functional as F


class MTSPEnv:
    """
    config: dict
        - salesmen  int
        - cities    int
            - salesmen <= cities;
        - mode ['fixed'|'rand'] str
        - seed      int
    """
    def __init__(self, config: Dict = None):
        self.cities = 50
        self.salesmen = 5
        self.seed = None
        self.mode = "rand"
        if config is not None:
            self.__parse_config(config)

        self.graph = None
        self.trajectories = None
        self.costs = None
        self.mask = None

        self.dim = 6
        self.step_count = 0
        self.step_limit = -1
        self.stay_still_limit = 3
        self.remain_stay_still_log = None

    def __parse_config(self, config: Dict):
        self.cities = config.get("cities", self.cities)
        self.salesmen = config.get("salesmen", self.salesmen)
        self.seed = config.get("seed", self.seed)
        self.mode = config.get("mode", self.mode)

        if self.seed is not None:
            np.random.seed(self.seed)
        self.GG = GG(1, self.cities, 2, self.seed)

    def __init(self, graph = None):
        if graph is not None:
            if len(graph.shape) ==2:
                self.graph = graph
            elif len(graph.shape) == 3:
                self.graph = graph[0]
            else:
                assert False
        elif self.graph is None or self.mode == "rand":
            self.graph = self.GG.generate(1,self.cities,2)[0]

        self.trajectories = [[1] for _ in range(self.salesmen)]
        self.costs = np.zeros(self.salesmen)
        self.mask = np.ones((self.cities,), dtype=np.float32)
        self.mask[0] = 0
        self.step_count = 0
        self.step_limit = self.cities // self.salesmen * 3
        self.remain_stay_still_log = np.zeros(self.salesmen, dtype=np.int32)

    def _get_salesman(self, idx):
        state = np.empty((self.dim,), dtype=np.float32)
        pos = self.trajectories[idx][-1]
        state[:2] = self.graph[0]
        state[2:4] = self.graph[pos-1]
        state[4] = self.costs[idx]
        state[5] = self._get_distance(1, pos)
        return state

    def _get_salesmen(self):
        states = np.empty((self.salesmen, self.dim), dtype=np.float32)
        for i in range(self.salesmen):
            states[i] = self._get_salesman(i)
        return states

    def _get_distance(self, id1, id2):
        return np.sqrt(np.sum(np.square(self.graph[id1-1] - self.graph[id2-1])))

    def _get_salesmen_masks(self):

        repeat_masks = self.mask[np.newaxis,].repeat(self.salesmen,axis=0)

        for i in range(self.salesmen):
            cur_pos = self.trajectories[i][-1]
            if self.remain_stay_still_log[i] < self.stay_still_limit:
                if cur_pos != 1 :
                    repeat_masks[i, cur_pos-1] = 1
            else:
                self.remain_stay_still_log[i] = 0

        return repeat_masks

    def reset(self, config = None, graph = None):
        if config is not None:
            self.__parse_config(config)
        self.__init(graph)

        env_info = {
            "graph": self.graph,
            "salesmen": self.salesmen,
            "cities": self.cities,
            "mask": self.mask,
            "salesmen_masks": self._get_salesmen_masks()
        }

        return self._get_salesmen(), env_info

    def __one_step(self, idx, act):

        if act == 0:
            return
        assert self.mask[act - 1] == 1, f"reach city '{act}' twice!"

        self.trajectories[idx].append(act)
        if act != 1:
            self.mask[act - 1] = 0
        self.costs[idx] += self._get_distance(self.trajectories[idx][-1], self.trajectories[idx][-2])


    def _get_reward(self):
        done = True
        for t in self.trajectories:
            if not (len(t) > 1 and t[-1] == 1):
                done = False
                break

        reward = 0

        if done:
            reward = -np.max(self.costs)

        if self.step_count >= self.step_limit:
            reward = -self.cities

        return reward

    def deal_conflict(self, actions:np.ndarray):
        mp = {}
        new_actions = np.zeros_like(actions)

        for idx, act in enumerate(actions):
            if act != 1:
                if act in mp:
                    mp[act].append(idx)
                else:
                    mp.update({act: [idx]})
            else:
                new_actions[idx] = act

        for act, idxs in mp.items():
            min_idx = idxs[0]
            min_cost = self.costs[min_idx] + self._get_distance(self.trajectories[min_idx][-1], act)
            for idx in idxs[1:]:
                nxt_cost = self.costs[idx] + self._get_distance(self.trajectories[idx][-1], act)
                if nxt_cost < min_cost:
                    min_cost = nxt_cost
                    min_idx = idx
            new_actions[min_idx] = act

        for idx in range(self.salesmen):
            cur_pos = self.trajectories[idx][-1]
            if cur_pos == new_actions[idx]:
                self.remain_stay_still_log[idx]+=1
                new_actions[idx] = 0

        return new_actions

    def step(self, actions: np.ndarray):

        actions = self.deal_conflict(actions)

        for i in range(self.salesmen):
            self.__one_step(i, actions[i])

        if np.all(self.mask == 0):
            self.mask[0] = 1

        reward = self._get_reward()
        done = reward != 0
        if done and reward < -np.max(self.costs):
            self.costs = np.array(-reward).repeat(self.salesmen, axis=0)

        if done:
            self.mask = np.zeros((self.cities,), dtype=np.float32)

        info = {
            "mask": self.mask,
            "salesmen_masks": self._get_salesmen_masks()
        }

        if done:
            info.update(
                {
                    "trajectories": self.trajectories,
                    "costs": self.costs,
                }
            )

        self.step_count+=1

        return self._get_salesmen(), reward, done, info

if __name__ == '__main__':
    cfg = {
        "salesmen": 3,
        "cities": 10,
        "seed": None,
        "mode": 'rand'
    }
    env = MTSPEnv(
        cfg
    )
    states, info = env.reset()
    graph = info["graph"]
    salesmen = info["salesmen"]
    cities = info["cities"]
    mask = info["mask"]

    done = False
    EndInfo = {}
    EndInfo.update(info)
    agent_config = {
        "city_nums": cities,
        "agent_nums": salesmen,
    }
    agent = RandomAgent(agent_config)
    agent.reset(agent_config)

    while not done:
        actions = agent.predict(states, mask, None)
        state, reward, done, info = env.step(actions)
        mask = info["mask"]
        if done:
            EndInfo.update(info)
    print(f"trajectory:{EndInfo}")

    gp = GP()
    gp.draw_route(graph, EndInfo["trajectories"], title="random", one_first=True)
    # env.draw_multi(graph,[1,2,3], [ EndInfo["trajectories"], EndInfo["trajectories"],EndInfo["trajectories"]],
    #                [0.1,0.2,0.3],["ss","sw","s22"],True)
