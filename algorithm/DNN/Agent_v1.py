import argparse

from model.model_v1 import Model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from utils.TensorTools import _convert_tensor
import numpy as np


class Agent:
    def __init__(self, args):
        self.args = args
        self.model = Model(agent_dim=args.agent_dim,
                           hidden_dim=args.hidden_dim,
                           embed_dim=args.embed_dim,
                           num_heads=args.num_heads,
                           num_layers=args.num_layers)
        self._gamma = args.gamma
        self.lr = args.lr
        self.grad_max_norm = args.grad_max_norm
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        self.device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def reset_graph(self, graph):
        """
        :param graph: [B,N,2]
        :return:
        """
        graph_t = _convert_tensor(graph, device=self.device, target_shape_dim=3)
        self.model.init_city(graph_t)
        # print(f"city_embed:{self.model.city_embed[0,0,:2]}")


    def __get_action_logprob(self, states, masks, mode="greedy"):
        actions_logits = self.model(states, masks)
        dist = torch.distributions.Categorical(logits=actions_logits)
        if mode == "greedy":
            action = torch.argmax(actions_logits, dim=-1)
            logprob = dist.log_prob(action)
        elif mode == "sample":
            action = dist.sample()
            logprob = dist.log_prob(action)
        else:
            raise NotImplementedError
        return action, logprob

    def predict(self, states_t, masks_t):
        self.model.train()
        actions, _ = self.__get_action_logprob(states_t, masks_t, mode="sample")
        return actions.cpu().numpy().squeeze(0)

    def exploit(self, states_t, masks_t, mode="greedy"):
        self.model.eval()
        actions, _ = self.__get_action_logprob(states_t, masks_t, mode=mode)
        return actions.cpu().numpy().squeeze(0)

    def get_cumulative_returns(self, reward_list):
        returns_numpy = np.zeros(len(reward_list))
        returns_numpy[-1] = reward_list[-1]
        for idx in range(-2, -len(reward_list) - 1, -1):
            returns_numpy[idx] = reward_list[idx] + self._gamma * returns_numpy[idx + 1]
        return returns_numpy

    def __update_net(self, optim, params, loss):
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, self.grad_max_norm)
        optim.step()

    def __get_logprob(self, states, masks, actions):
        actions_logits = self.model(states, masks)
        dist = torch.distributions.Categorical(logits=actions_logits)
        entropy = dist.entropy()
        return dist.log_prob(actions), entropy

    def __get_loss(self, states, masks, actions, returns):
        actions_logprob, entropy = self.__get_logprob(states, masks, actions)

        if self.args.returns_norm:
            loss = - (actions_logprob * (returns - returns.mean()) / (returns.std() + 1e-8)).mean()
        else:
            loss = - (actions_logprob * returns).mean()

        if self.args.max_ent:
            loss -= self.args.entropy_coef * entropy.mean()

        return loss

    def learn(self, graph_t, states_tb, actions_tb, returns_tb, masks_tb):
        self.model.train()
        self.model.init_city(graph_t)
        # print(f"city_embed:{self.model.city_embed[0,0,:2]}")
        loss = self.__get_loss(states_tb, masks_tb, actions_tb, returns_tb)
        self.__update_net(self.optim, self.model.parameters(), loss)
        self.model.init_city(graph_t)
        # print(f"city_embed:{self.model.city_embed[0, 0, :2]}")
        return loss.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_max_norm", type=float, default=0.5)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--returns_norm", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=1e-2)
    args = parser.parse_args()

    def eval(env, agent, graph_plot, graph, min_reward):
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
        if reward > min_reward:
            graph_plot.draw_route(graph, info["actors_trajectory"], title=f"agent_cost:{-reward}_time:{used_time}", one_first=True)
            min_reward = reward
        return min_reward
    from envs.GraphGenerator import GraphGenerator as GG

    graphG = GG(1, 10, 2)
    graph = graphG.generate()
    from envs.MTSP.MTSP import MTSPEnv

    env = MTSPEnv()
    env.init_fixed_graph(graph, 2)

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
        reward= 0
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
        returns_tb = torch.tensor(np.repeat(returns_numpy[:, np.newaxis], a_num, axis=1), dtype=torch.float32, device=device)
        masks_tb = torch.cat(masks_list, dim=0)
        loss = agent.learn(torch.tensor(graph, dtype=torch.float32, device= device), [last_states_tb, states_tb], actions_tb, returns_tb, masks_tb)
        if i % 100 == 0:
            print(f"loss:{loss}, reward:{reward}")
            eval_reward = eval(env, agent, graph_plot, graph, min_reward)
            min_reward = eval_reward
