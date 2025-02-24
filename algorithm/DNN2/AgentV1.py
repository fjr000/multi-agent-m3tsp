import argparse
import time

from model.nModel.model_v1 import Model
import torch
from utils.TensorTools import _convert_tensor
import numpy as np
from algorithm.DNN2.AgentBase import AgentBase
import tqdm


class AgentV1(AgentBase):
    def __init__(self, args):
        super(AgentV1, self).__init__(args,Model)
        self.model.to(self.device)

    def save_model(self, id):
        filename = f"nAgentV1_{id}"
        super(AgentV1, self)._save_model(self.args.model_dir, filename)

    def load_model(self, id):
        filename = f"nAgentV1_{id}"
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

    graphG = GG(1, 50, 2)
    graph = graphG.generate()
    from envs.MTSP.MTSP import MTSPEnv

    env = MTSPEnv()
    from algorithm.OR_Tools.mtsp import ortools_solve_mtsp

    indexs, cost, used_time = ortools_solve_mtsp(graph, args.agent_num, 10000)
    env.draw(graph, cost, indexs, used_time, agent_name="or_tools")
    print(f"or tools :{cost}")
    agent = AgentV1(args)
    min_greedy_cost = 1000
    min_sample_cost = 1000
    loss_list = []
    for i in tqdm.tqdm(range(100_000_000)):
        features_nb, actions_nb, returns_nb, masks_nb, dones_nb = agent.run_batch(env, graph, args.agent_num, args.batch_size)
        loss = agent.learn(_convert_tensor(graph, dtype=torch.float32, device=agent.device, target_shape_dim=3),
                           _convert_tensor(features_nb,dtype = torch.float32, device=agent.device),
                           _convert_tensor(actions_nb,dtype = torch.float32, device=agent.device),
                           _convert_tensor(returns_nb,dtype = torch.float32, device=agent.device),
                           _convert_tensor(masks_nb,dtype = torch.float32, device=agent.device),
                           dones_nb
                           )
        loss_list.append(loss)
        if i % 10 == 0:
            print(f"loss:{np.mean(np.array(loss_list))}")
            loss_list.clear()

            st =time.time_ns()
            greedy_cost, greedy_trajectory = agent.eval_episode(env, graph, args.agent_num, exploit_mode="greedy")
            ed = time.time_ns()
            if greedy_cost < min_greedy_cost:
                env.draw(graph, greedy_cost, greedy_trajectory, (ed - st) * 1e-9)
                min_greedy_cost = greedy_cost
            print(f"eval greedy cost:{greedy_cost}, min greedy cost:{min_greedy_cost}")
            epoch_min_sample_cost = 1000
            min_sample_trajectory = None
            for i in range(64):
                st = time.time_ns()
                sample_cost, sample_trajectory = agent.eval_episode(env, graph, args.agent_num)
                ed = time.time_ns()
                if sample_cost < epoch_min_sample_cost:
                    epoch_min_sample_cost = sample_cost
                    min_sample_trajectory = sample_trajectory

            if epoch_min_sample_cost < min_sample_cost:
                min_sample_cost = epoch_min_sample_cost
                env.draw(graph, epoch_min_sample_cost, min_sample_trajectory, (ed - st) * 1e-9)
            print(f"eval sample cost:{epoch_min_sample_cost}, min sample cost:{min_sample_cost}")