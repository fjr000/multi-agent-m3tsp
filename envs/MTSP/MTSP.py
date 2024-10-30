import numpy as np
import gym
from typing import Tuple, List
import sys
import os

sys.path.append("../")
from Config import Config
from envs.GraphGenerator import GraphGenerator as GG
from utils.GraphPlot import GraphPlot as GP

class MTSPEnv:
    """
    Limitations::
    - Can't move backward (action == 1) before traveling
    - Can't reach the depot before all cities have been visited
    - All salesmen must arrive at the depot at the same time (at the end).
    - Salesmen can't visit the same city except for the depot.
    """
    def __init__(self, city_nums: Tuple = (10, 100), agent_nums: Tuple = (1, 9), seed=None):
        """

        :param city_nums: (min_num, max_num); min_num <= max_num
        :param agent_nums: (min_num, max_num); min_num <= max_num; agent_num < city_num
        """
        assert isinstance(city_nums, Tuple) or isinstance(city_nums, List), "city_nums must be a tuple or list"
        assert isinstance(agent_nums, Tuple) or isinstance(agent_nums, List), "agent_nums must be a tuple or list"
        assert len(city_nums) == 2 and len(agent_nums) == 2, "city_nums and agent_nums must have equal length 2"
        assert city_nums[0] <= city_nums[1], "city_min_num must be less than or equal to city_max_num"
        assert agent_nums[0] <= agent_nums[1], "agent_min_num must be less than or equal to agent_max_num"
        assert city_nums[0] > agent_nums[1], "agent_min_num must be greater than city_max_num"
        self.city_nums = city_nums
        self.agent_nums = agent_nums
        self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)
        self.GG = GG(1, self.city_nums[1], 2, seed)
        self.actual_city_num = np.random.randint(self.city_nums[0], self.city_nums[1] +1)
        self.actual_agent_num = np.random.randint(self.agent_nums[0], self.agent_nums[1]+1)
        self.city_dims = 2
        self.agent_dims = 3

        self.init_graph = None
        self.init_graph_matrix = None
        self.global_mask = None
        self.actors_state = None
        self.actors_trajectory = None
        self.actors_action_record = None
        self.actors_cost = None
        self.travel_over = None
        self.actors_action_mask = None
        self.to_end = False
        self.reset_state()

    def reset_state(self):
        self.init_graph = None
        self.init_graph_matrix = None
        self.global_mask = np.zeros(self.actual_city_num)
        self.global_mask[0] = 1
        self.actors_action_mask = np.zeros((self.actual_agent_num,2))
        self.actors_action_mask[:,0] = 1
        self.actors_state = np.zeros((self.actual_agent_num, self.agent_dims), dtype=np.float32)
        self.actors_trajectory = [[1] for _ in range(self.actual_agent_num)]
        self.actors_action_record = [[] for _ in range(self.actual_agent_num)]
        self.actors_cost = np.zeros(self.actual_agent_num, dtype=np.float32)
        self.travel_over = np.zeros(self.actual_agent_num, dtype=np.bool_)
        self.to_end = False

    def reset(self, city_nums: Tuple = None, agent_nums: Tuple = None, fixed_graph=False, seed=None):
        """

        :param seed: random seed
        :param fixed_graph: whether return fixed graph or not
        :param city_nums: (min_num, max_num); min_num <= max_num
        :param agent_nums: (min_num, max_num); min_num <= max_num; agent_num < city_num
        """

        if fixed_graph:
            if self.init_graph is None:
                self.reset_state()
                self.__init_graph()
        else:
            if city_nums is not None:
                assert isinstance(city_nums, Tuple) or isinstance(city_nums, List), "city_nums must be a tuple or list"
                assert len(city_nums) == 2, "city_nums must have equal length 2"
                assert city_nums[0] <= city_nums[1], "city_min_num must be less than or equal to city_max_num"
                self.city_nums = city_nums

            if agent_nums is not None:
                assert isinstance(agent_nums, Tuple) or isinstance(agent_nums,
                                                                   List), "agent_nums must be a tuple or list"
                assert len(agent_nums) == 2, "city_nums and agent_nums must have equal length 2"
                assert agent_nums[0] <= agent_nums[1], "agent_min_num must be less than or equal to agent_max_num"
                assert self.city_nums[0] > agent_nums[1], "agent_min_num must be greater than city_max_num"
                self.agent_nums = agent_nums

            if self.seed is not None:
                np.random.seed(self.seed)
                self.GG = GG(1, self.city_nums[1], self.city_dims, seed)

            self.actual_city_num = np.random.randint(self.city_nums[0], self.city_nums[1]+1)
            self.actual_agent_num = np.random.randint(self.agent_nums[0], self.agent_nums[1]+1)
            self.reset_state()
            self.__init_graph()

        return self.actors_state, \
            {
                "cnum": self.actual_city_num,
                "anum": self.actual_agent_num,
                "graph": self.init_graph,
                "graph_matrix": self.init_graph_matrix,
                "global_mask": self.global_mask,
                "agents_action_mask": self.get_agents_action_mask(),
            }

    def __init_graph(self):
        self.init_graph = self.GG.generate(1, self.actual_city_num, self.city_dims)
        self.init_graph_matrix = self.GG.nodes_to_matrix(self.init_graph).squeeze(0)
        self.init_graph = self.init_graph.squeeze(0)
        for agent_id in range(self.actual_agent_num):
            self.__update_agent_state(agent_id)

    def step(self, actions: np.ndarray):
        """
        :param actions: m actions
        :return: observation, reward, done, info
        """
        assert actions is not None and isinstance(actions, np.ndarray), "actions must be a numpy array"
        assert actions.dtype == np.int32 or actions.dtype == np.int64, "action must be int type"
        assert actions.shape[-1] == self.actual_agent_num, "actions must have shape equal to agent_nums"
        assert np.max(actions) <= self.actual_city_num, "actions"

        for agent_id, action in enumerate(actions):
            self.one_step(agent_id, action)

        done = self.is_done()
        reward = self.get_reward()
        if self.to_end and not done:
            self.global_mask[0] = 0

        info = {
                "global_mask": self.global_mask,
                "agents_action_mask": self.get_agents_action_mask(),
            }

        if done:
            info.update({
                "actors_cost": self.actors_cost,
                "actors_trajectory": self.actors_trajectory,
                "actors_action_record": self.actors_action_record,
            })
        return self.actors_state, reward, done, info

    def one_step(self, agent_id: int, action: int):
        """
        -1: back previous city
        0:  stay
        1:  depot
        1<=a<=city_num : city index
        :param agent_id: 0< agent id < self.actual_agent_num
        :param action: [-1, self.actual_city_num]
        :return: done
        """
        assert 0 <= agent_id <= self.actual_agent_num, "agent_id out of range"
        assert -1 <= action <= self.actual_city_num, "action out of range"

        if action == -1:
            assert len(self.actors_trajectory[agent_id]) > 1, "can't back before travel"
            last_cost = self.get_cost(self.actors_trajectory[agent_id][-1], self.actors_trajectory[agent_id][-2])
            if self.actors_trajectory[agent_id][-1] != 1:
                self.global_mask[self.actors_trajectory[agent_id][-1]-1] = 0
            self.actors_trajectory[agent_id].pop()
            self.actors_cost[agent_id] -= last_cost
        elif action != 0:
            if action != 1:
                assert self.global_mask[action-1] == 0, f"reach city {action} twice"
            cost = self.get_cost(self.actors_trajectory[agent_id][-1], action)
            self.actors_trajectory[agent_id].append(action)
            self.actors_cost[agent_id] += cost
            self.global_mask[action-1] = 1

        self.__update_agent_state(agent_id)
        self.actors_action_record[agent_id].append(action)

    def get_city_cord(self, city_id: int):
        """
        city id is in [1,city_num]
        :param city_id:
        :return: city_id's cord [x,y]
        """
        return self.init_graph[city_id - 1]

    def get_cost(self, city_id1: int, city_id2: int):
        """
        city id is in [1,city_num]
        :param city_id1:
        :param city_id2:
        :return: cost between city_id1 and city_id2
        """
        assert self.init_graph_matrix is not None, "init_graph_matrix must not be None"
        return self.init_graph_matrix[city_id1 - 1][city_id2 - 1]

    def __update_agent_state(self, agent_id: int):
        cur_city_id = self.actors_trajectory[agent_id][-1]
        cur_city_cord = self.get_city_cord(cur_city_id)
        self.actors_state[agent_id, :2] = cur_city_cord
        self.actors_state[agent_id, 2] = self.actors_cost[agent_id]

    def get_agent_state(self, agent_id):
        return self.actors_state[agent_id]

    def __is_to_end(self):
        self.to_end = np.count_nonzero(self.global_mask) == self.actual_city_num
        return self.to_end

    def is_done(self):
        self.__is_to_end()
        if self.to_end:
            for idx in range(self.actual_agent_num):
                if self.actors_trajectory[idx][-1] != 1:
                    return False
            return True
        return False

    def get_reward(self, is_done: bool = False):
        if is_done:
            return -np.max(self.actors_cost)
        else:
            return 0

    def __update_agent_back_mask(self, agent_id: int):
        self.actors_action_mask[agent_id, 0] = (self.actors_trajectory[agent_id][-1] == 1) or self.to_end

    def get_agents_action_mask(self):
        self.actors_action_mask[:, 1] = self.to_end
        for i in range(self.actual_agent_num):
            self.__update_agent_back_mask(i)
        return self.actors_action_mask

    @staticmethod
    def final_action_choice(action_to_chose, action_mask):
        special_action_to_chose = np.where(action_mask == 0)[0] - 1
        return np.concatenate((special_action_to_chose, action_to_chose), axis=-1)

if __name__ == '__main__':
    env = MTSPEnv(city_nums=(5, 10), agent_nums=(2, 3))
    states, info = env.reset()
    anum = info["anum"]
    cnum = info["cnum"]
    graph = info["graph"]
    global_mask = info["global_mask"]
    agents_action_mask = info["agents_action_mask"]

    done = False
    EndInfo = {}
    while not done:
        action_to_chose = np.where(global_mask == 0)[0] + 1
        actions = np.zeros(anum, dtype=np.int32)
        for i in range(anum):
            final_action_to_chose = MTSPEnv.final_action_choice(action_to_chose, agents_action_mask[i])
            actions[i] = np.random.choice(final_action_to_chose)
            if actions[i] != 0 and actions[i] != -1:
                if actions[i] != 1:
                    idx = np.where(action_to_chose == actions[i])[0]
                    action_to_chose = np.delete(action_to_chose, idx)
        # action = np.random.choice(action_to_chose, anum, replace=False)
        state, reward, done, info = env.step(actions)
        global_mask = info["global_mask"]
        agents_action_mask = info["agents_action_mask"]
        if done:
            EndInfo.update(info)
    print(f"trajectory:{EndInfo}")
    gp = GP()
    gp.draw_route(graph,EndInfo["actors_trajectory"],title="random", one_first=True)