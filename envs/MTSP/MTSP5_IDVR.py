import argparse
import copy

from envs.MTSP.MTSP5 import MTSPEnv
import numpy as np

class MTSPEnv_IDVR(MTSPEnv):
    def __init__(self, config):
        super(MTSPEnv_IDVR, self).__init__(config)
        self.last_costs = None

    def _init(self, graph=None):
        super(MTSPEnv_IDVR,self)._init(graph)
        self.last_costs = copy.deepcopy(self.costs)
        self.last_potential_reward = self.advanced_minmax_potential()
        self.team_reward = None

    def advanced_minmax_potential(self):
        """针对MINMAXMTSP优化的潜在函数"""
        unvisited_cities_num = np.count_nonzero(self.mask[...,1:], axis = -1, keepdims=True) / (self.cities - 1)
        current_costs = self.costs

        # 计算当前路径长度
        max_path_length = np.max(current_costs, axis = -1, keepdims = True)
        avg_path_length = self.costs / self.path_count

        # 潜在值计算
        potential = -(
                unvisited_cities_num * 1.0 +  # 未访问城市惩罚
                max_path_length * 0.25 +
                self.costs / (1e-8 + max_path_length) * 1.0 +
                avg_path_length * 1.0
        )

        return potential

    def _get_reward(self):
        self.dones = np.all(self.stage_2, axis=1)
        self.done = np.all(self.dones, axis=0)
        # # self.rewards = np.where(self.dones[:,None], -np.max(self.costs,keepdims=True, axis=1).repeat(self.salesmen, axis = 1), 0)
        # # # self.rewards = self.last_costs - self.costs
        # # self.rewards = self.last_costs -np.max(self.costs,keepdims=True, axis=1).repeat(self.salesmen, axis = 1)
        # # self.last_costs = copy.deepcopy(np.max(self.costs,keepdims=True, axis=1).repeat(self.salesmen, axis = 1))
        #
        # """计算基于潜在函数的奖励塑形"""
        # # 计算当前状态的潜在值
        # current_potential = self.advanced_minmax_potential()
        #
        # # 计算塑形奖励
        # shaping_reward = 0.99 * current_potential - self.last_potential_reward
        # self.last_potential_reward = current_potential
        # self.rewards = shaping_reward
        self.rewards = self.last_costs - self.costs
        self.team_reward = np.max(self.last_costs, axis = 1) - np.max(self.costs, axis = 1)

        self.last_costs = self.costs.copy()


        return self.rewards, self.team_reward

    def step(self, action):
        states, r, d, info = super(MTSPEnv_IDVR,self).step(action)
        info.update({
            'team_reward': self.team_reward,
        })
        return states, r, d, info

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
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad_max_norm", type=float, default=1)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=5e-3)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--city_nums", type=int, default=50)
    parser.add_argument("--random_city_num", type=bool, default=True)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=0)
    parser.add_argument("--env_masks_mode", type=int, default=0,
                        help="0 for only the min cost  not allow back depot; 1 for only the max cost allow back depot")
    parser.add_argument("--eval_interval", type=int, default=500, help="eval  interval")
    parser.add_argument("--use_conflict_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_conflict_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_actions_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_city_encoder", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--use_agents_mask", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--use_city_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--agents_adv_rate", type=float, default=0.2, help="rate of adv between agents")
    parser.add_argument("--conflict_loss_rate", type=float, default=0.5 + 0.5, help="rate of adv between agents")
    parser.add_argument("--only_one_instance", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--save_model_interval", type=int, default=10000, help="save model interval")
    parser.add_argument("--seed", type=int, default=528, help="random seed")
    args = parser.parse_args()

    env_config = {
        "salesmen": args.agent_num,
        "cities": args.city_nums,
        "seed": None,
        "mode": 'rand',
        "env_masks_mode":args.env_masks_mode,
        "use_conflict_model": args.use_conflict_model
    }
    env = MTSPEnv_IDVR(
        env_config
    )

    from envs.GraphGenerator import GraphGenerator as GG

    # g = GG(1, env_config["cities"])
    # graph = g.generate(1, env_config["cities"], dim=2)

    from algorithm.DNN5.AgentIDVR import AgentIDVR as Agent
    from model.n4Model.config import Config as Config

    agent = Agent(args, Config)
    agent.load_model(args.agent_id)
    # features_nb, actions_nb, actions_no_conflict_nb, returns_nb, individual_returns_nb, masks_nb, dones_nb = agent.run_batch(
    #     env, graph, env_config["salesmen"], 32)
    from utils.TensorTools import _convert_tensor
    import numpy as np
    import torch
    from envs.GraphGenerator import GraphGenerator as GG
    g =GG()
    batch_graph = g.generate(batch_size=args.batch_size,num=args.city_nums)
    states, info = env.reset(graph = batch_graph)
    salesmen_masks = info["salesmen_masks"]
    agent.reset_graph(batch_graph,3)
    done =False
    agent.run_batch_episode(env, batch_graph, args.agent_num, False, info={
                "use_conflict_model": args.use_conflict_model})