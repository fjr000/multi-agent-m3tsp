import argparse
import time

from model.SeqModel.model_v1 import Model
import torch
from utils.TensorTools import _convert_tensor
import numpy as np
from algorithm.SeqAgent.AgentBase import AgentBase
import tqdm


class AgentV1(AgentBase):
    def __init__(self, args, config):
        super(AgentV1, self).__init__(args, config, Model)
        self.model.to(self.device)

    def save_model(self, id):
        filename = f"SeqAgentV1_{id}"
        super(AgentV1, self)._save_model(self.args.model_dir, filename)

    def load_model(self, id):
        filename = f"SeqAgentV1_{id}"
        super(AgentV1, self)._load_model(self.args.model_dir, filename)


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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_max_norm", type=float, default=1)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=1e-2)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--city_nums", type=int, default=50)
    parser.add_argument("--random_city_num", type=bool, default=True)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=0)
    parser.add_argument("--env_masks_mode", type=int, default=4,
                        help="0 for only the min cost  not allow back depot; 1 for only the max cost allow back depot")
    parser.add_argument("--eval_interval", type=int, default=400, help="eval  interval")
    parser.add_argument("--use_conflict_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_conflict_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_actions_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_city_encoder", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--use_agents_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--use_city_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--agents_adv_rate", type=float, default=0.5, help="rate of adv between agents")
    parser.add_argument("--conflict_loss_rate", type=float, default=0.1, help="rate of adv between agents")
    parser.add_argument("--only_one_instance", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--save_model_interval", type=int, default=10000, help="save model interval")
    parser.add_argument("--seed", type=int, default=528, help="random seed")
    args = parser.parse_args()

    from envs.GraphGenerator import GraphGenerator as GG

    graphG = GG(args.batch_size, 50, 2)
    graph = graphG.generate()
    from envs.MTSP.MTSP5_IDVR import MTSPEnv_IDVR as MTSPEnv

    env = MTSPEnv({
        "env_masks_mode": args.env_masks_mode,
    })
    from algorithm.OR_Tools.mtsp import ortools_solve_mtsp

    # indexs, cost, used_time = ortools_solve_mtsp(graph, args.agent_num, 10000)
    # env.draw(graph, cost, indexs, used_time, agent_name="or_tools")
    # print(f"or tools :{cost}")
    from model.n4Model.config import Config

    agent = AgentV1(args, Config)
    min_greedy_cost = 1000
    min_sample_cost = 1000
    loss_list = []
    buffer = agent.run_batch_episode(
        env, graph, args.agent_num, eval_mode=False, exploit_mode="sample"
    )

    policy_loss, value_loss, entropy_loss = agent.learn(*buffer)
