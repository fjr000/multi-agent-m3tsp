from pydoc_data.topics import topics
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
    def graph2matrix(graph: torch.Tensor) -> torch.Tensor:
        coords_expanded_1 = graph.unsqueeze(2)
        coords_expanded_2 = graph.unsqueeze(1)
        distance_matrix = torch.sqrt(torch.sum((coords_expanded_1 - coords_expanded_2) ** 2, dim=-1))
        return distance_matrix

    @staticmethod
    def init(graph: torch.Tensor, n_agent: int) -> 'States':
        batch, n_city, _ = graph.shape

        virtual_graph = torch.cat([
            graph[:, 0:1, :].repeat(1, n_agent, 1),
            graph
        ], dim=1)

        mask = torch.cat([
            torch.zeros((batch, n_agent + 1), dtype=torch.bool, device=graph.device),
            torch.ones((batch, n_city - 1), dtype=torch.bool, device=graph.device),
        ], dim=1)

        if n_agent > 1:
            mask[..., 1] = True

        matrix = States.graph2matrix(virtual_graph)
        max_distance = torch.max(matrix[:, 0, :], dim=-1, keepdim=True)[0]

        return States(
            n_agent=n_agent,
            n_city=n_city,
            graph=graph,
            virtual_graph=virtual_graph,
            matrix=matrix,
            step=0,
            mask=mask,
            costs=torch.zeros((batch, n_agent), dtype=torch.float32, device=graph.device),
            remain_n_city=torch.ones((batch, 1), dtype=torch.int32, device=graph.device) * (n_city - 1),
            remain_n_agent=torch.ones((batch, 1), dtype=torch.int32, device=graph.device) * n_agent,
            visited=torch.zeros((batch, n_agent + n_city), dtype=torch.int32, device=graph.device),
            remain_city_ratio=torch.ones((batch, 1), dtype=torch.float32, device=graph.device) * (n_city - 1) / n_city,
            remain_agent_ratio=torch.ones((batch, 1), dtype=torch.float32, device=graph.device),
            max_distance=max_distance,
            remain_max_distance=max_distance,
            agent_id=torch.zeros((batch, 1), dtype=torch.int32, device=graph.device),
            cur_pos=torch.zeros((batch, 1), dtype=torch.int32, device=graph.device),
            dis_depot=torch.zeros((batch, 1), dtype=torch.float32, device=graph.device),
        )

    def update(self, selected: torch.Tensor) -> 'States':
        '''
        Args:
            selected: [B, 1]
        Returns: states
        '''
        B = selected.shape[0]
        batch_indices = torch.arange(B, device=self.mask.device)
        step = self.step + 1
        selected_sqz = selected.squeeze(-1)
        switch_ = selected == (self.agent_id + 1)  # batch needs to switch to next agent

        # Check if all selected cities are valid
        x= torch.argwhere(~self.mask[batch_indices, selected_sqz])
        assert torch.all(self.mask[batch_indices, selected_sqz]), "visited city twice!"

        # Update visited cities
        visited = self.visited.clone()
        visited[:, step:step + 1] = selected

        # Update mask
        mask = self.mask.clone()
        mask[batch_indices, selected_sqz] = False

        # Update costs
        costs = self.costs.clone()
        costs[batch_indices, self.agent_id.squeeze(-1)] += \
            self.matrix[batch_indices, self.cur_pos.squeeze(-1), selected_sqz]

        # Update remaining cities and agents
        remain_n_city = torch.where(switch_, self.remain_n_city, self.remain_n_city - 1)
        remain_n_agent = torch.where(switch_, self.remain_n_agent - 1, self.remain_n_agent)

        # Update agent ID
        agent_id = torch.where(switch_, self.agent_id + 1, self.agent_id)

        # Update mask for next agent's starting position
        allow = ((remain_n_agent > 1) | ((remain_n_agent == 1) & (remain_n_city == 0)))
        mask[batch_indices, agent_id.squeeze(-1) + 1] = allow.squeeze(-1)

        # Update current position
        cur_pos = torch.where(switch_, torch.zeros_like(selected), selected)

        # Update distance to depot
        dis_depot = self.matrix[batch_indices, cur_pos.squeeze(-1), 0].unsqueeze(-1)

        # Update remaining max distance
        remain_max_distance = self.matrix[:,0,:].clone()
        remain_max_distance = torch.where(self.mask, remain_max_distance, -torch.inf)
        remain_max_distance,_ = torch.max(remain_max_distance, dim=-1, keepdim=True)

        return self._replace(
            step=step,
            visited=visited,
            mask=mask,
            costs=costs,
            remain_n_city=remain_n_city,
            remain_n_agent=remain_n_agent,
            remain_city_ratio=remain_n_city.float() / self.n_city,
            remain_agent_ratio=remain_n_agent.float() / self.n_agent,
            remain_max_distance=remain_max_distance,
            agent_id=agent_id,
            dis_depot=dis_depot,
            cur_pos=cur_pos,
        )

    def n2t(self, device):
        return self._replace(
            mask=~self.mask.clone(),
        )

    def t2n(self):
        return self._replace(
            n_agent=n_agent,
            n_city=n_city,
            graph=graph,
            virtual_graph=self.virtual_graph.cpu().numpy(),
            matrix=self.matrix.cpu().numpy(),
            step=0,
            mask=~self.mask.cpu().numpy(),
            costs=self.costs.cpu().numpy(),
            remain_n_city=self.remain_n_city.cpu().numpy(),
            remain_n_agent=self.remain_n_agent.cpu().numpy(),
            visited=self.visited.cpu().numpy(),
            remain_city_ratio=self.remain_city_ratio.cpu().numpy(),
            remain_agent_ratio=self.remain_agent_ratio.cpu().numpy(),
            max_distance=self.max_distance.cpu().numpy(),
            remain_max_distance=self.remain_max_distance.cpu().numpy(),
            agent_id=self.agent_id.cpu().numpy(),
            cur_pos=self.cur_pos.cpu().numpy(),
            dis_depot=self.dis_depot.cpu().numpy(),
        )

    def _to_trajs(self):
        trajs = []
        for b in range(self.costs.shape[0]):
            traj = []
            traja = [1]
            for x in self.visited[b]:
                x = x.item()
                if x == 0:
                    continue
                elif x <= self.n_agent:
                    traja.append(1)
                    traj.append(traja)
                    traja = [1]
                else:
                    traja.append(x - self.n_agent + 1)
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
        self.states = States.init(_convert_tensor(batch_graph,device=torch.device("cuda:0")), n_agent)
        return self.states

    def _get_reward(self):
        x,_ = self.states.costs.max(dim=1)
        return -x  # 最大行程作为reward

    def step(self, action):
        self.states = self.states.update(_convert_tensor(action,dtype=torch.long,device=torch.device("cuda:0")))

        reward = self._get_reward()

        done = (self.states.remain_n_agent == 0).all() & (self.states.remain_n_city == 0).all()

        if done:
            info = {
                "costs" : self.states.costs.cpu(),
                "trajectories" : self.states.t2n()
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
