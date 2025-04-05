import argparse
import time

from model.n4Model.model_v4 import Model
import torch
from utils.TensorTools import _convert_tensor
import numpy as np
from algorithm.DNN5.AgentBase import AgentBase
import tqdm


class AgentV4(AgentBase):
    def __init__(self, args, config):
        super(AgentV4, self).__init__(args, config, Model)
        self.model.to(self.device)

    def save_model(self, id):
        filename = f"n5AgentV4_{id}"
        super(AgentV4, self)._save_model(self.args.model_dir, filename)

    def load_model(self, id):
        filename = f"n5AgentV4_{id}"
        super(AgentV4, self)._load_model(self.args.model_dir, filename)

    def _get_loss(self, act_logp, agents_logp, costs):

        # 智能体间平均， 组间最小化最大
        costs_8 = costs.reshape(costs.shape[0] // 8, 8, -1)  # 将成本按实例组进行分组
        act_logp_8 = act_logp.reshape(act_logp.shape[0] // 8, 8, -1)  # 将动作概率按实例组进行分组

        # agents_avg_cost = np.mean(costs_8, keepdims=True, axis=-1)
        agents_max_cost = np.max(costs_8, keepdims=True, axis=-1)
        # 智能体间优势
        agents_adv = costs_8 - agents_max_cost
        # agents_adv = agents_adv - agents_adv.mean(keepdims=True, axis=-1)
        agents_adv = (agents_adv - agents_adv.mean(keepdims=True, axis=-1)) / (
                    agents_adv.std(axis=-1, keepdims=True) + 1e-8)
        # agents_adv = (agents_adv - agents_adv.mean( keepdims=True,axis = -1))/(agents_adv.std(axis=-1, keepdims=True) + 1e-8)
        # 实例间优势
        # group_adv = agents_max_cost - np.mean(agents_max_cost, keepdims=True, axis=1)
        group_adv = (agents_max_cost - np.mean(agents_max_cost, keepdims=True, axis=1)) / (
                    agents_max_cost.std(keepdims=True, axis=1) + 1e-8)
        # 组合优势
        adv = self.args.agents_adv_rate*agents_adv + group_adv

        # 转换为tensor并放到指定的device上
        adv_t = _convert_tensor(adv, device=self.device)

        # 对动作概率为零的样本进行掩码
        mask_ = ((act_logp_8 != 0) & (~torch.isnan(act_logp_8)))

        # 计算动作网络的损失，mask之后加权平均
        act_loss = (act_logp_8[mask_] * adv_t[mask_]).mean()
        if agents_logp is not None:
            agents_logp_8 = agents_logp.reshape(agents_logp.shape[0] // 8, 8, -1)  # 将智能体动作概率按实例组进行分组
            # 对智能体的动作概率进行掩码
            mask_ = ((agents_logp_8 != 0) & (~torch.isnan(agents_logp_8)))

            # 计算智能体的损失，mask之后加权平均
            agents_loss = (agents_logp_8[mask_] * adv_t[mask_]).mean()
        else:
            agents_loss = None

        return act_loss, agents_loss

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