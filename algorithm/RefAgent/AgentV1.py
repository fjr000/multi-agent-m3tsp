import argparse
import time

from model.RefModel.Model import Model
from algorithm.DNN5.AgentBase import AgentBase


class AgentV1(AgentBase):
    def __init__(self, args, config):
        super(AgentV1, self).__init__(args, config, Model)
        self.model.to(self.device)

    def save_model(self, id):
        filename = f"RefAgentV1_{id}"
        super(AgentV1, self)._save_model(self.args.model_dir, filename)

    def load_model(self, id):
        filename = f"RefAgentV1_{id}"
        super(AgentV1, self)._load_model(self.args.model_dir, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_num", type=int, default=5)
    parser.add_argument("--agent_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_max_norm", type=float, default=1.0)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--returns_norm", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=float, default=256)
    parser.add_argument("--model_dir", type=str, default="../../pth/")
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
    from model.n4Model.config import Config

    agent = AgentV1(args, Config)
    min_greedy_cost = 1000
    min_sample_cost = 1000
    loss_list = []
    act_logp, agents_logp, costs = agent.run_batch_episode(env, graph, args.agent_num, eval_mode=False,
                                                           exploit_mode="sample")

    act_loss, agents_loss = agent.learn(act_logp, agents_logp, costs)