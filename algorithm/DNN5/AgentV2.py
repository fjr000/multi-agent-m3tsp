import argparse
import time

from model.n4Model.model_v3 import Model
import torch
from utils.TensorTools import _convert_tensor
import numpy as np
from algorithm.DNN5.AgentBase import AgentBase
import tqdm


class AgentV2(AgentBase):
    def __init__(self, args, config):
        super(AgentV2, self).__init__(args, config, Model)
        self.model.to(self.device)

    def save_model(self, id):
        filename = f"n5AgentV2_{id}"
        super(AgentV2, self)._save_model(self.args.model_dir, filename)

    def load_model(self, id):
        filename = f"n5AgentV2_{id}"
        super(AgentV2, self)._load_model(self.args.model_dir, filename)
        

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
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--grad_max_norm", type=float, default=10)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=1e-2)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--city_nums", type=int, default=40)
    parser.add_argument("--random_city_num", type=bool, default=True)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=80000)
    parser.add_argument("--env_masks_mode", type=int, default=1,
                        help="0 for only the min cost  not allow back depot; 1 for only the max cost allow back depot")
    parser.add_argument("--eval_interval", type=int, default=100, help="eval  interval")
    parser.add_argument("--use_conflict_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--only_one_instance", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--save_model_interval", type=int, default=10000, help="save model interval")
    args = parser.parse_args()

    from envs.GraphGenerator import GraphGenerator as GG

    graphG = GG(args.batch_size, 50, 2)
    graph = graphG.generate()
    from envs.MTSP.MTSP4 import MTSPEnv

    env = MTSPEnv()
    from algorithm.OR_Tools.mtsp import ortools_solve_mtsp

    # indexs, cost, used_time = ortools_solve_mtsp(graph, args.agent_num, 10000)
    # env.draw(graph, cost, indexs, used_time, agent_name="or_tools")
    # print(f"or tools :{cost}")
    from model.n4Model.model_v3 import Config
    agent = AgentV2(args, Config)
    min_greedy_cost = 1000
    min_sample_cost = 1000
    loss_list = []
    act_logp, agents_logp, act_ent, agt_ent, costs = agent.run_batch_episode(env, graph, args.agent_num,
                                                                             eval_mode=False, info={
            "use_conflict_model": args.use_conflict_model})
    act_loss, agents_loss, act_ent_loss, agt_ent_loss = agent.learn(act_logp, agents_logp, act_ent, agt_ent, costs)
