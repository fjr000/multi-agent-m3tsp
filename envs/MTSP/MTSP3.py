import argparse
import copy
import math

import numpy as np
from typing import Tuple, List, Dict
import sys

sys.path.append("../")
from envs.MTSP.Config import Config
from envs.GraphGenerator import GraphGenerator as GG
from utils.GraphPlot import GraphPlot as GP
from model.NNN.RandomAgent import RandomAgent
import torch.nn.functional as F


class MTSPEnv:
    """
    1. 智能体离开仓库后，再次抵达仓库后不允许其离开仓库，salesman mask中只有depot
    2. 允许智能体中当前旅行成本最高M-1个返回仓库，由最后一个智能体保证完备性（还是通过允许最远的返回仓库？）
    3. 允许智能体静止在原地（限制与否？）

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
        self.last_costs = None
        self.costs = None
        self.mask = None

        self.dim = 14
        self.step_count = 0
        self.step_limit = -1
        self.stay_still_limit = -1
        self.remain_stay_still_log = None

        self.traj_stages = None
        self.dones_step = None
        self.salesmen_masks = None
        self.actions = None
        self.ori_actions = None

        self.distance_scale = np.sqrt(2)

    def __parse_config(self, config: Dict):
        self.cities = config.get("cities", self.cities)
        self.salesmen = config.get("salesmen", self.salesmen)
        self.seed = config.get("seed", self.seed)
        self.mode = config.get("mode", self.mode)

        if self.seed is not None:
            np.random.seed(self.seed)
        self.GG = GG(1, self.cities, 2, self.seed)

    def __init(self, graph=None):
        if graph is not None:
            if len(graph.shape) == 2:
                self.graph = graph
            elif len(graph.shape) == 3:
                self.graph = graph[0]
            else:
                assert False
        elif self.graph is None or self.mode == "rand":
            self.graph = self.GG.generate(1, self.cities, 2)[0]

        self.trajectories = [[1] for _ in range(self.salesmen)]
        self.last_costs = np.zeros(self.salesmen)
        self.costs = np.zeros(self.salesmen)
        self.mask = np.ones((self.cities,), dtype=np.float32)
        self.mask[0] = 0
        self.step_count = 0
        self.step_limit = self.cities
        self.remain_stay_still_log = np.zeros(self.salesmen, dtype=np.int32)
        self.traj_stages = np.zeros(self.salesmen,
                                    dtype=np.int32)  # 0 -> prepare; 1 -> travelling; 2 -> finished; 3 -> stay depot
        self.dones_step = np.zeros(self.salesmen, dtype=np.int32)

        self.salesmen_masks = None
        self.actions = None
        self.ori_actions = None

        self.distance_scale = self.cities / self.salesmen

    def _get_salesman(self, idx):
        state = np.empty((self.dim,), dtype=np.float32)
        pos = self.trajectories[idx][-1]

        state[0] = 0  # depot indice
        state[1] = pos - 1  # cur indice

        state[2] = self._get_distance(1, pos) / self.distance_scale  # distance from depot
        state[3] = self.costs[idx] / self.distance_scale  # cur cost scale
        state[4] = np.max(self.costs) / self.distance_scale  # max cost
        state[5] = (state[4] - state[3]) / (state[4] + 1e-8) # diff cost from max cost
        state[6] = np.min(self.costs) / self.distance_scale # min cost
        state[7] = (state[6] - state[3]) / (state[4] + 1e-8) # diff cost from min cost
        state[8] = np.mean(
            [self._get_distance(pos, i+1)
             for i in range(self.cities)
             if self.salesmen_masks[idx,i] > 0.5]
        )/self.distance_scale

        remain_salesmen_num = np.count_nonzero(self.traj_stages < 2)
        remain_cities_num = np.count_nonzero(self.salesmen_masks[idx])
        state[9] = remain_salesmen_num / self.salesmen  # remain agents ratio
        state[10] = remain_cities_num / self.cities  # remain city ratio
        state[11] = remain_salesmen_num / (remain_cities_num + 1e-8)

        state[12] = np.argsort(self.costs)[idx] / self.salesmen  # rank
        state[13] = np.sum(self.costs - self.costs[idx]) / (self.salesmen-1) / self.distance_scale

        return state

    def _get_salesmen(self):
        states = np.empty((self.salesmen, self.dim), dtype=np.float32)
        for i in range(self.salesmen):
            states[i] = self._get_salesman(i)
        if np.any(np.isnan(states)):
            pass
        return states

    def _get_distance(self, id1, id2):
        return np.sqrt(np.sum(np.square(self.graph[id1 - 1] - self.graph[id2 - 1])))

    def _get_salesmen_masks(self):

        # init salesmen masks form global mask
        repeat_masks = self.mask[np.newaxis,].repeat(self.salesmen, axis=0)

        # decide whether salesmen can stay still
        for i in range(self.salesmen):
            cur_pos = self.trajectories[i][-1]
            if self.remain_stay_still_log[i] < self.stay_still_limit:
                if cur_pos != 1:
                    repeat_masks[i, cur_pos - 1] = 1

        # set traj finished mask [1,0,...]
        for i in range(self.salesmen):
            if self.traj_stages[i] >= 2:
                repeat_masks[i] = np.zeros((self.cities,), dtype=np.float32)
            if self.traj_stages[i] >= 1:
                repeat_masks[i, 0] = 1

        # decide whether salesmen can return depot
        if np.any(self.mask):
            # allow return depot instead of the min cost trajectory
            costs = np.where(self.traj_stages >= 2, np.inf, self.costs)
            min_cost_id = np.argmin(costs)
            repeat_masks[min_cost_id, 0] = 0

        self.salesmen_masks = repeat_masks

        return repeat_masks

    def reset(self, config=None, graph=None):
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

        if not (self.mask[act - 1] == 1 or act == 1):
            pass

        assert self.mask[act - 1] == 1 or act == 1, f"reach city '{act}' twice!"

        self.trajectories[idx].append(act)
        if act != 1:
            self.mask[act - 1] = 0
            self.traj_stages[idx] = 1
        else:
            self.traj_stages[idx] = 2
            self.dones_step[idx] = self.step_count
        self.costs[idx] += self._get_distance(self.trajectories[idx][-1], self.trajectories[idx][-2])

    def _get_individual_rewards(self, actions):
        rewards = np.zeros(self.salesmen, dtype=np.float32)
        for i in range(self.salesmen):
            if actions[i] == 0:
                rewards[i] = -0.001
                if self.traj_stages[i] >= 2:
                    rewards[i] = 0
            else:
                rewards[i] = -self._get_distance(self.trajectories[i][-1], self.trajectories[i][-2])
                if self.traj_stages[i] >= 2:
                    self.traj_stages[i] += 1

        self.individual_rewards = rewards
        return rewards

    def _get_individual_rewards2(self, actions):
        rewards = np.zeros(self.salesmen, dtype=np.float32)
        if np.all(self.traj_stages >= 2):
            max_cost = np.max(self.costs)
            rewards += - max_cost               * 1.0
            rewards += (max_cost - self.costs)  * 0.05
            rewards += - self.costs             * 0.05
            rewards /= max_cost
        else:
            remain_city_num = np.count_nonzero(self.mask)
            rewards += np.where( np.logical_and(self.traj_stages>=2 , actions == 1), -remain_city_num / self.cities, 0)
        # else:
        #     max_cost = np.max(self.costs)
        #     last_max_cost = np.max(self.last_costs)
        #     rewards += 0.1 * ((- self.last_costs + last_max_cost) - (- self.costs + max_cost))
        self.individual_rewards = rewards
        return rewards

    def _get_reward(self):
        done = True
        for t in self.trajectories:
            if not (len(t) > 1 and t[-1] == 1):
                done = False
                break

        reward = 0

        if done:
            reward = -np.max(self.costs)

        if self.step_count > self.step_limit:
            reward = -self.cities

        return reward

    def deal_conflict(self, actions: np.ndarray):
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

        return new_actions

    def reset_actions(self, actions):
        for idx in range(self.salesmen):
            cur_pos = self.trajectories[idx][-1]
            if cur_pos == actions[idx]:
                self.remain_stay_still_log[idx] += 1
                actions[idx] = 0
            else:
                self.remain_stay_still_log[idx] = 0

        return actions

    def step(self, actions: np.ndarray):

        self.step_count += 1
        self.last_costs = copy.deepcopy(self.costs)

        self.ori_actions = copy.deepcopy(actions)
        actions = self.reset_actions(actions)
        # actions = self.deal_conflict(actions)
        self.actions = actions
        for i in range(self.salesmen):
            self.__one_step(i, actions[i])

        # if np.all(self.mask == 0):
        #     self.mask[0] = 1
        individual_rewards = self._get_individual_rewards2(actions)
        reward = self._get_reward()
        done = reward != 0
        if done and reward < -np.max(self.costs):
            self.costs = np.array(-reward).repeat(self.salesmen, axis=0)

        if done:
            self.mask = np.zeros((self.cities,), dtype=np.float32)

        info = {
            "mask": self.mask,
            "salesmen_masks": self._get_salesmen_masks(),
            "individual_rewards": individual_rewards,
            "dones_step": self.dones_step,
        }

        if done:
            info.update(
                {
                    "trajectories": self.trajectories,
                    "costs": self.costs,
                }
            )

        return self._get_salesmen(), reward, done, info

    def draw(self, graph, cost, trajectory, used_time=0, agent_name="agent", draw=True):
        from utils.GraphPlot import GraphPlot as GP
        graph_plot = GP()
        if agent_name == "or_tools":
            one_first = False
        else:
            one_first = True
        return graph_plot.draw_route(graph, trajectory, draw=draw,
                                     title=f"{agent_name}_cost:{cost:.5f}_time:{used_time:.3f}", one_first=one_first)

    def draw_multi(self, graph, costs, trajectorys, used_times=(0,), agent_names=("agents",), draw=True):
        figs = []
        for c, t, u, a in zip(costs, trajectorys, used_times, agent_names):
            figs.append(self.draw(graph, c, t, u, a, False))
        from utils.GraphPlot import GraphPlot as GP
        graph_plot = GP()
        fig = graph_plot.combine_figs(figs)
        if draw:
            fig.show()
        return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_worker", type=int, default=2)
    parser.add_argument("--agent_num", type=int, default=5)
    parser.add_argument("--agent_dim", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_max_norm", type=float, default=10)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--returns_norm", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=float, default=512)
    parser.add_argument("--city_nums", type=int, default=50)
    parser.add_argument("--allow_back", type=bool, default=False)
    parser.add_argument("--model_dir", type=str, default="../../pth/")
    parser.add_argument("--agent_id", type=int, default=0)
    args = parser.parse_args()

    env_config = {
        "salesmen": args.agent_num,
        "cities": args.city_nums,
        "seed": None,
        "mode": 'rand'
    }
    env = MTSPEnv(
        env_config
    )

    from envs.GraphGenerator import GraphGenerator as GG

    g = GG(1, env_config["cities"])
    graph = g.generate(1, env_config["cities"], dim=2)

    from algorithm.DNN4.AgentV2 import AgentV2 as Agent
    from model.n4Model.config import Config

    agent = Agent(args, Config)
    agent.load_model(args.agent_id)
    features_nb, actions_nb, actions_no_conflict_nb, returns_nb, individual_returns_nb, masks_nb, dones_nb = agent.run_batch(
        env, graph, env_config["salesmen"], 32)
    from utils.TensorTools import _convert_tensor
    import numpy as np
    import torch

    loss = agent.learn(_convert_tensor(graph, dtype=torch.float32, device=agent.device, target_shape_dim=3),
                       _convert_tensor(features_nb, dtype=torch.float32, device=agent.device),
                       _convert_tensor(actions_nb, dtype=torch.float32, device=agent.device),
                       # _convert_tensor(returns_nb, dtype=torch.float32, device=train_agent.device),
                       _convert_tensor(individual_returns_nb, dtype=torch.float32, device=agent.device),
                       # rewards_nb,
                       _convert_tensor(masks_nb, dtype=torch.float32, device=agent.device),
                       dones_nb,
                       # _convert_tensor(logp_nb, dtype=torch.float32, device=train_agent.device),
                       )
    # gp = GP()
    # gp.draw_route(graph, eval_info[1], title=f"costs:{np.max(eval_info[0])},{np.min(eval_info[0])}", one_first=True)
    # # env.draw_multi(graph,[1,2,3], [ EndInfo["trajectories"], EndInfo["trajectories"],EndInfo["trajectories"]],
    # #                [0.1,0.2,0.3],["ss","sw","s22"],True)
