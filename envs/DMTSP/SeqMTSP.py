from typing import Dict, NamedTuple, Union
import numpy as np
import sys
import torch
from utils.TensorTools import _convert_tensor

sys.path.append("../../")
sys.path.append("../")

class States(NamedTuple):
    # global
    graph:  Union[np.ndarray, torch.Tensor] # [B, N, 2]
    matrix: Union[np.ndarray, torch.Tensor] # [B, N, N]
    step :  int # [0, N+M)
    mask :  Union[np.ndarray, torch.Tensor] # [B, N]
    costs:  Union[np.ndarray, torch.Tensor] # [B, M]
    remain_n_city:  Union[np.ndarray, torch.Tensor] # [B, 1]
    remain_n_agent: Union[np.ndarray, torch.Tensor] # [B, 1]
    visited: Union[np.ndarray, torch.Tensor]    # [B, N+M]

    # priv
    agent_id:   Union[np.ndarray, torch.Tensor]     # [B, 1]
    cur_pos:    Union[np.ndarray, torch.Tensor]     # [B, 1]
    dis_depot:  Union[np.ndarray, torch.Tensor]     # [B, 1]

    @staticmethod
    def graph2matrix(graph):
        coords_expanded_1 = np.expand_dims(graph, axis=2)
        coords_expanded_2 = np.expand_dims(graph, axis=1)
        distance_matrix = np.sqrt(np.sum((coords_expanded_1 - coords_expanded_2) ** 2, axis=-1))
        return distance_matrix

    @staticmethod
    def init(graph, n_agent):
        batch, n_city, _ = graph.shape

        graph = np.concatenate([
            graph[:,0:1,:].repeat(n_agent, axis=1),
            graph
        ], axis=1)

        mask = np.concatenate([
            np.zeros((batch, n_agent+1), dtype=np.bool_),
            np.ones((batch, n_city-1), dtype=np.bool_),
        ], axis=1)
        mask[..., 1] = True

        return States(
            graph=graph,
            matrix=States.graph2matrix(graph),
            step=0,
            mask=mask,
            costs= np.zeros((batch, n_agent), dtype=np.float32),
            remain_n_city=np.ones((batch, 1), dtype=np.long) * (n_city -1),
            remain_n_agent=np.ones((batch, 1), dtype=np.long) * n_agent,
            visited=np.zeros((batch, n_agent + n_city), dtype=np.long),

            agent_id=np.zeros((batch, 1), dtype=np.long),
            cur_pos=np.zeros((batch, 1), dtype=np.long),
            dis_depot=np.zeros((batch, 1), dtype=np.float32),
        )

    def update(self, selected):
        '''
        Args:
            selected: [B, 1]
        Returns: states
        '''

        # precompute
        B = selected.shape[0]
        batch_indices1d = np.arange(B)
        step = self.step + 1
        selected_sqz = selected.squeeze(-1)
        switch_ = selected == self.agent_id + 1 # batch need switch next agent

        assert np.all(self.mask[batch_indices1d, selected_sqz]), "visited city twice !"
        self.visited[:, step:step+1] = selected
        self.mask[batch_indices1d, selected_sqz] = False

        self.costs[batch_indices1d, self.agent_id.squeeze(-1)]\
            +=self.matrix[batch_indices1d, self.cur_pos.squeeze(-1), selected_sqz]

        remain_n_city = np.where(switch_, self.remain_n_city, self.remain_n_city-1)
        remain_n_agent = np.where(switch_, self.remain_n_agent-1, self.remain_n_agent)

        agent_id = np.where(switch_, self.agent_id+1, self.agent_id)

        allow = ((remain_n_agent >1) | ((remain_n_agent == 1) & (remain_n_city == 0)))
        self.mask[batch_indices1d, agent_id.squeeze(-1)+1] = allow.squeeze(-1)
        cur_pos = np.where(switch_, 0, selected)
        dis_depot = self.matrix[batch_indices1d, cur_pos.squeeze(-1), 0][...,None]


        return self._replace(
            step = step,
            remain_n_city = remain_n_city,
            remain_n_agent = remain_n_agent,
            agent_id = agent_id,
            dis_depot = dis_depot,
            cur_pos = cur_pos,
        )

    def n2t(self, device):
        return self._replace(
            mask=_convert_tensor(~self.mask, dtype=torch.bool, device=device),
            costs=_convert_tensor(self.costs, dtype=torch.float32, device=device),
            remain_n_city=_convert_tensor(self.remain_n_city, dtype=torch.float32, device=device),
            remain_n_agent=_convert_tensor(self.remain_n_agent, dtype=torch.float32, device=device),
            agent_id=_convert_tensor(self.agent_id, dtype=torch.long, device=device),
            cur_pos=_convert_tensor(self.cur_pos, dtype=torch.long, device=device),
            dis_depot=_convert_tensor(self.dis_depot, dtype=torch.float32, device=device)
        )

class SeqMTSPEnv:
    def __init__(self, seed = None):
        self.n_city = None
        self.n_salesmen = None
        self.b_graph = None
        self.n_graph = None

        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

    @staticmethod
    def generate_graph(size = 1, n_city = 50):
        return np.random.uniform(0, 1, (size, n_city, 2))

    def reset(self, batch_graph, n_salesmen):
        self.n_graph, self.n_city, _ = batch_graph.shape
        self.b_graph = batch_graph
        self.n_salesmen = n_salesmen


if __name__ == '__main__':
    n_city = 10
    n_agent = 3
    n_graph = 2

    graph = SeqMTSPEnv.generate_graph(n_graph, n_city)

    state = States.init(graph, n_agent)

    selecteds = np.array([
        [[5], [12]],
        [[1], [11]],
        [[2], [10]],
        [[6], [1]],
        [[7], [9]],
        [[9], [2]],
        [[8], [8]],
        [[4], [5]],
        [[11], [4]],
        [[12], [6]],
        [[10], [7]],
        [[3], [3]],
    ])
    for x in selecteds:
        state = state.update(x)
        pass


