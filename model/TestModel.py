import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.ActionReSelector import ActionReSelectorBySeq as ARS
import Net

class TestModel(nn.Module):
    def __init__(self, agent_state_dim, city_state_dim,  embedding_dim = 128):
        super(TestModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.city_state_dim = city_state_dim
        self.agent_state_dim = agent_state_dim

        self.agents_encoder = Net.SelfEncoder([self.agent_state_dim],self.embedding_dim,8)
        self.cities_encoder = Net.SelfEncoder([self.city_state_dim for _ in range(3)],self.embedding_dim,8)

        self.cities_embedding = None
        self.graph = None


        self.ARS = ARS()

    def initGraph(self, graph, depot_idx = 1):
        self.graph = graph
        self.cities_embedding = self.cities_encoder()


    def forward(self, agent_state, city_state, global_mask, agent_action_mask = None):
        assert agent_state.shape[-1] == self.agent_state_dim, f"the dimension of input agent state is {agent_state.shape[-1]} and should be equal to {self.agent_state_dim} "
        assert city_state.shape[-1] == self.city_state_dim, f"the dimension of input agent state is {city_state.shape[-1]} and should be equal to {self.city_state_dim} "
