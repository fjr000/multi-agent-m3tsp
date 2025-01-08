import argparse

from model.model_v1 import Model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from utils.TensorTools import _convert_tensor
import numpy as np
from algorithm.DNN.AgentBase import AgentBase


class Agent(AgentBase):
    def __init__(self, args):
        super(Agent, self).__init__(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_max_norm", type=float, default=0.5)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--returns_norm", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=1e-2)
    args = parser.parse_args()


    def eval(env, agent, graph_plot, graph, min_reward):
        import time
        st = time.time_ns()
        agents_states, info = env.reset({"fixed_graph": True})
        agents_mask = info["agents_mask"]
        agents_last_states = info["actors_last_states"]
        done = False
        device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
        agent.reset_graph(graph)
        reward = 0
        info = None
        while not done:
            agents_states_t = _convert_tensor(agents_states, device=device, target_shape_dim=3)
            last_states_t = _convert_tensor(agents_last_states, device=device, target_shape_dim=3)
            agents_mask_t = _convert_tensor(agents_mask, device=device, target_shape_dim=3)
            actions = agent.exploit([last_states_t, agents_states_t], agents_mask_t)
            states, reward, done, info = env.step(actions + 1)
            agents_mask = info["agents_mask"]
            agents_last_states = info["actors_last_states"]
            agents_states = states
        print(f"eval {reward}")
        ed = time.time_ns()
        if reward > min_reward:
            graph_plot.draw_route(graph, info["actors_trajectory"],
                                  title=f"agent_cost:{-reward}_time:{1e-9 * (ed - st)}", one_first=True)
            min_reward = reward
        return min_reward


    from envs.GraphGenerator import GraphGenerator as GG

    graphG = GG(1, 20, 2)
    graph = graphG.generate()
    from envs.MTSP.MTSP import MTSPEnv

    env = MTSPEnv()
    env.init_fixed_graph(graph, 3)

    agents_states, info = env.reset({"fixed_graph": True})
    a_num = info["anum"]

    from algorithm.OR_Tools.mtsp import ortools_solve_mtsp

    indexs, cost, used_time = ortools_solve_mtsp(graph, a_num, 10000)
    from utils.GraphPlot import GraphPlot as GP

    graph_plot = GP()
    graph_plot.draw_route(graph, indexs, title=f"or_tools_cost:{cost}_time:{used_time}")
    print(f"or tools :{cost}")
    agent = Agent(args)
    min_reward = -1000
    states_tb_list = []
    states_tb_list = []
    last_states_tb_list = []
    actions_tb_list = []
    returns_tb_list = []
    masks_tb_list = []
    sample_len = 0
    loss = 0

    for i in range(100_000_000):
        agents_states, info = env.reset({"fixed_graph": True})
        agents_mask = info["agents_mask"]
        agents_last_states = info["actors_last_states"]
        a_num = info["anum"]

        done = False
        device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
        agent.reset_graph(graph)
        last_states_list = []
        states_list = []
        actions_list = []
        reward_list = []
        masks_list = []
        reward = 0
        while not done:
            agents_states_t = _convert_tensor(agents_states, device=device, target_shape_dim=3)
            last_states_t = _convert_tensor(agents_last_states, device=device, target_shape_dim=3)
            agents_mask_t = _convert_tensor(agents_mask, device=device, target_shape_dim=3)
            actions = agent.predict([last_states_t, agents_states_t], agents_mask_t)

            states, reward, done, info = env.step(actions + 1)

            last_states_list.append(last_states_t)
            states_list.append(agents_states_t)
            masks_list.append(agents_mask_t)
            actions_list.append(torch.tensor(actions, dtype=torch.float32, device=device).unsqueeze(0))
            reward_list.append(reward)

            agents_mask = info["agents_mask"]
            agents_last_states = info["actors_last_states"]
            agents_states = states

        returns_numpy = agent.get_cumulative_returns(reward_list)
        states_tb = torch.cat(states_list, dim=0)
        last_states_tb = torch.cat(last_states_list, dim=0)
        actions_tb = torch.cat(actions_list, dim=0)
        returns_tb = torch.tensor(np.repeat(returns_numpy[:, np.newaxis], a_num, axis=1), dtype=torch.float32,
                                  device=device)
        masks_tb = torch.cat(masks_list, dim=0)

        states_tb_list.append(states_tb)
        last_states_tb_list.append(last_states_tb)
        actions_tb_list.append(actions_tb)
        returns_tb_list.append(returns_tb)
        masks_tb_list.append(masks_tb)
        sample_len += states_tb.size(0)
        if sample_len > 1024:
            loss = agent.learn(torch.tensor(graph, dtype=torch.float32, device=device),
                               [torch.cat(last_states_tb_list, dim=0), torch.cat(states_tb_list, dim=0)],
                               torch.cat(actions_tb_list, dim=0),
                               torch.cat(returns_tb_list, dim=0),
                               torch.cat(masks_tb_list, dim=0)
                               )
            sample_len = 0
            states_tb_list.clear()
            actions_tb_list.clear()
            last_states_tb_list.clear()
            masks_tb_list.clear()
            returns_tb_list.clear()
        if i % 100 == 0:
            print(f"loss:{loss}, reward:{reward}")
            eval_reward = eval(env, agent, graph_plot, graph, min_reward)
            min_reward = eval_reward
