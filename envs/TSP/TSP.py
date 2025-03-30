import argparse

import gym
import numpy as np
from click.core import batch

from utils.EvalTools import EvalTools
from envs.GraphGenerator import GraphGenerator as GG
from envs.MTSP.MTSP4 import MTSPEnv

class TSPEnv(MTSPEnv):
    def __init__(self, config={}):
        super(TSPEnv, self).__init__(config)
        self.graph_padded = None
        self.mask_padded = None
        self.traj_len = None
        self.trajs = None
        self.B = None
        self.A = None
        self.max_len_traj = None
        self.env_masks_mode = 1
        self.map = None

    def reset(self, config=None, graph=None):
        """
        Args:
            graph: np.ndarray [B, N, 2]
            trajs: list [B, A, L], L为每个智能体序列的长度 每个L的长度可能不同，

        Returns:

        """
        assert config is not None
        trajs = config.get("trajs", None)
        assert trajs is not None

        self._parse_config(config)

        self.B, N, _ = graph.shape
        self.A = len(trajs[0])
        B =self.B
        A = self.A
        self.traj_len = np.array([len(a) for b in trajs for a in b])-1
        self.max_len_traj = max(self.traj_len)
        for b in range(B):
            for a in range(A):
                diff = self.max_len_traj  - len(trajs[b][a]) +1
                trajs[b][a] += [1] * diff
        self.trajs = np.array(trajs)

        self.salesmen = 1
        self.cities = self.max_len_traj

        # 3. 填充数据
        batch_indices = np.arange(B)[..., None, None]
        self.graph_padded = graph[batch_indices, self.trajs[...,:-1]-1, :]  # shape [B, A, L, 2]
        self.mask_padded =  np.where(self.trajs[...,:-1] == 1, False, True)

        self.graph_padded = self.graph_padded.reshape(B*A, self.max_len_traj, 2)
        self.mask_padded = self.mask_padded.reshape(B*A, self.max_len_traj)
        self._init(self.graph_padded)

        self.mask = self.mask_padded

        env_info = {
            "graph": self.graph_padded,
            "salesmen": self.salesmen,
            "cities": self.cities,
            "mask": self.mask,
            "salesmen_masks": self._get_salesmen_masks(),
            "masks_in_salesmen": self._get_masks_in_salesmen(),
        }

        return self._get_salesmen_states(), env_info

    def step(self, action):
        states, rewards, done, info = super(TSPEnv, self).step(action)
        if done:
            trajs = info['trajectories'].reshape(self.B,self.A,-1)
            costs = info['costs'].reshape(self.B,self.A)

            info['trajectories'] = np.take_along_axis(self.trajs, trajs-1, axis=-1)
            info['costs'] = costs
        return states, rewards, done, info




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--agent_num", type=int, default=10)
    parser.add_argument("--fixed_agent_num", type=bool, default=False)
    parser.add_argument("--agent_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad_max_norm", type=float, default=1)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=1e-2)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--city_nums", type=int, default=50)
    parser.add_argument("--random_city_num", type=bool, default=False)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=140000)
    parser.add_argument("--env_masks_mode", type=int, default=4,
                        help="0 for only the min cost  not allow back depot; 1 for only the max cost allow back depot")
    parser.add_argument("--eval_interval", type=int, default=400, help="eval  interval")
    parser.add_argument("--use_conflict_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_conflict_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_actions_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_city_encoder", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--use_agents_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--use_city_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--agents_adv_rate", type=float, default=0.1, help="rate of adv between agents")
    parser.add_argument("--conflict_loss_rate", type=float, default=0.1, help="rate of adv between agents")
    parser.add_argument("--only_one_instance", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--save_model_interval", type=int, default=10000, help="save model interval")
    parser.add_argument("--seed", type=int, default=528, help="random seed")
    args = parser.parse_args()

    tsp = TSPEnv({})
    from envs.MTSP.MTSP5 import MTSPEnv
    from model.n4Model.config import Config

    batch_size = 10

    gGG = GG(batch_size=batch_size, num=10)
    batch_graph = gGG.generate()
    from algorithm.DNN5.AgentV4 import AgentV4 as Agent
    import sys
    sys.path.append('../../')
    sys.path.append('../')
    sys.path.append('../../pth')
    agent = Agent(args,Config)
    agent.load_model(args.agent_id)
    mtsp = MTSPEnv({"env_masks_mode":args.env_masks_mode})
    greedy_cost, traj, greedy_time = EvalTools.EvalGreedy(batch_graph, 2, agent, mtsp)
    tsp_cost, tsp_traj, tsp_time = EvalTools.EvalTSPGreedy(batch_graph, agent, traj)
    for i  in range(batch_size):
        mtsp.draw_multi(batch_graph[i], [greedy_cost[i], tsp_cost[i]],[traj[i], tsp_traj[i]], [0,0],["g","tsp"],True )
    pass
