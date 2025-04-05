from typing import Dict, NamedTuple, Union
import numpy as np
import sys
import torch
from sympy.physics.units.definitions.dimension_definitions import information

from utils.TensorTools import _convert_tensor

sys.path.append("../../")
sys.path.append("../")


class States(NamedTuple):
    # global
    n_city: int
    n_agent: int
    graph: Union[np.ndarray, torch.Tensor]  # [B, N, 2]
    virtual_graph: Union[np.ndarray, torch.Tensor]  # [B, N, 2]
    matrix: Union[np.ndarray, torch.Tensor]  # [B, N, N]
    step: int  # [0, N+M)
    mask: Union[np.ndarray, torch.Tensor]  # [B, N]
    costs: Union[np.ndarray, torch.Tensor]  # [B, M]
    remain_n_city: Union[np.ndarray, torch.Tensor]  # [B, 1]
    remain_n_agent: Union[np.ndarray, torch.Tensor]  # [B, 1]
    visited: Union[np.ndarray, torch.Tensor]  # [B, N+M]

    remain_city_ratio: Union[np.ndarray, torch.Tensor]
    remain_agent_ratio: Union[np.ndarray, torch.Tensor]

    max_distance : Union[np.ndarray, torch.Tensor]
    remain_max_distance: Union[np.ndarray, torch.Tensor]

    # priv
    agent_id: Union[np.ndarray, torch.Tensor]  # [B, 1]
    cur_pos: Union[np.ndarray, torch.Tensor]  # [B, 1]
    dis_depot: Union[np.ndarray, torch.Tensor]  # [B, 1]

    @staticmethod
    def graph2matrix(graph):
        coords_expanded_1 = np.expand_dims(graph, axis=2)
        coords_expanded_2 = np.expand_dims(graph, axis=1)
        distance_matrix = np.sqrt(np.sum((coords_expanded_1 - coords_expanded_2) ** 2, axis=-1))
        return distance_matrix

    @staticmethod
    def init(graph, n_agent):
        batch, n_city, _ = graph.shape

        virtual_graph = np.concatenate([
            graph[:, 0:1, :].repeat(n_agent, axis=1),
            graph
        ], axis=1)

        mask = np.concatenate([
            np.zeros((batch, n_agent + 1), dtype=np.bool_),
            np.ones((batch, n_city - 1), dtype=np.bool_),
        ], axis=1)
        if n_agent >1:
            mask[..., 1] = True
        matrix = States.graph2matrix(virtual_graph)
        max_distance = np.max(matrix[:,0,:],axis=-1, keepdims=True)
        return States(
            n_agent=n_agent,
            n_city=n_city,
            graph=graph,
            virtual_graph=virtual_graph,
            matrix=matrix,
            step=0,
            mask=mask,
            costs=np.zeros((batch, n_agent), dtype=np.float32),
            remain_n_city=np.ones((batch, 1), dtype=np.int32) * (n_city - 1),
            remain_n_agent=np.ones((batch, 1), dtype=np.int32) * n_agent,
            visited=np.zeros((batch, n_agent + n_city), dtype=np.int32),
            remain_city_ratio=np.ones((batch, 1), dtype=np.float32) * (n_city - 1) / n_city,
            remain_agent_ratio=np.ones((batch, 1), dtype=np.float32),

            max_distance=max_distance,
            remain_max_distance=max_distance,

            agent_id=np.zeros((batch, 1), dtype=np.int32),
            cur_pos=np.zeros((batch, 1), dtype=np.int32),
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
        switch_ = selected == self.agent_id + 1  # batch need switch next agent

        assert np.all(self.mask[batch_indices1d, selected_sqz]), "visited city twice !"
        self.visited[:, step:step + 1] = selected
        self.mask[batch_indices1d, selected_sqz] = False

        self.costs[batch_indices1d, self.agent_id.squeeze(-1)] \
            += self.matrix[batch_indices1d, self.cur_pos.squeeze(-1), selected_sqz]

        remain_n_city = np.where(switch_, self.remain_n_city, self.remain_n_city - 1)
        remain_n_agent = np.where(switch_, self.remain_n_agent - 1, self.remain_n_agent)

        agent_id = np.where(switch_, self.agent_id + 1, self.agent_id)

        allow = ((remain_n_agent > 1) | ((remain_n_agent == 1) & (remain_n_city == 0)))
        self.mask[batch_indices1d, agent_id.squeeze(-1) + 1] = allow.squeeze(-1)
        cur_pos = np.where(switch_, 0, selected)
        dis_depot = self.matrix[batch_indices1d, cur_pos.squeeze(-1), 0][..., None]
        remain_max_distance = self.matrix[:,0,:]
        remain_max_distance = np.where(self.mask, remain_max_distance, np.nan)
        remain_max_distance = np.max(remain_max_distance, axis=-1, keepdims=True)

        return self._replace(
            step=step,
            remain_n_city=remain_n_city,
            remain_n_agent=remain_n_agent,
            remain_city_ratio=remain_n_city / self.n_city,
            remain_agent_ratio=remain_n_agent / self.n_agent,
            remain_max_distance=remain_max_distance,
            agent_id=agent_id,
            dis_depot=dis_depot,
            cur_pos=cur_pos,
        )

    def n2t(self, device):
        return self._replace(
            graph=_convert_tensor(self.graph, dtype=torch.float32, device=device),
            mask=_convert_tensor(~self.mask, dtype=torch.bool, device=device),
            costs=_convert_tensor(self.costs, dtype=torch.float32, device=device),
            remain_city_ratio=_convert_tensor(self.remain_city_ratio, dtype=torch.float32, device=device),
            remain_agent_ratio=_convert_tensor(self.remain_agent_ratio, dtype=torch.float32, device=device),
            agent_id=_convert_tensor(self.agent_id, dtype=torch.long, device=device),
            cur_pos=_convert_tensor(self.cur_pos, dtype=torch.long, device=device),
            max_distance=_convert_tensor(self.max_distance, dtype=torch.float32, device=device),
            remain_max_distance=_convert_tensor(self.remain_max_distance, dtype=torch.float32, device=device),
        )

    def _to_trajs(self):
        trajs = []
        for b in range(self.costs.shape[0]):
            traj = []
            traja = [1]
            for x in self.visited[b]:
                if x == 0:
                    continue
                elif x <= self.n_agent:
                    traja.append(1)
                    traj.append(traja)
                    traja = [1]
                else:
                    traja.append(x.item()-self.n_agent +1)
            trajs.append(traj)
        return trajs

class SeqMTSPEnv:
    def __init__(self, seed=None):
        self.n_city = None
        self.n_agent = None
        self.b_graph = None
        self.n_graph = None

        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

        self.states = None

    @staticmethod
    def generate_graph(size=1, n_city=50):
        return np.random.uniform(0, 1, (size, n_city, 2))

    def reset(self, batch_graph, n_agent):
        self.n_graph, self.n_city, _ = batch_graph.shape
        self.b_graph = batch_graph
        self.n_agent = n_agent
        self.states = States.init(batch_graph, n_agent)
        return self.states

    def _get_reward(self):
        return -self.states.costs.max(axis=1)  # 最大行程作为reward

    def step(self, action):
        self.states = self.states.update(action)

        reward = self._get_reward()

        done = (self.states.remain_n_agent == 0).all() & (self.states.remain_n_city == 0).all()

        if done:
            info = {
                "costs" : self.states.costs,
                "trajectories" : self.states
            }
        else:
            info = {}
        return self.states, reward, done, info

    def draw_multi(self, graph, costs, trajectorys, used_times=(0,), agent_names=("agents",), draw=True):
        figs = []
        for c, t, u, a in zip(costs, trajectorys, used_times, agent_names):
            figs.append(self.draw(graph, c, t, u, a, False))
        from utils.GraphPlot import GraphPlot as GP
        graph_plot = GP()
        fig = graph_plot.combine_figs(figs)
        if draw:
            fig.show()
        return fig

if __name__ == '__main__':
    n_city = 10
    n_agent = 3
    n_graph = 2

    graph = SeqMTSPEnv.generate_graph(n_graph, n_city)
    env = SeqMTSPEnv(seed=0)
    # state = States.init(graph, n_agent)
    state = env.reset(graph, n_agent)

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
    s,r,d,info = None,None,None, None
    for x in selecteds:
        # state = state.update(x)
        s,r,d,info=env.step(x)
        pass
    cost = info['costs']
    def dis(i,x,y):
        return np.sum((graph[i,x] - graph[i,y])**2)**0.5
    c = 0
    p = 0
    cc = []
    for x in range(len(selecteds)):
        if selecteds[x,0,0] <=n_agent:
            c+=dis(0,p,0)
            p=0
            cc.append(c)
            c=0
        else:
            c+=dis(0,p,selecteds[x,0,0]-n_agent)
            p = selecteds[x,0,0]-n_agent

    pass
