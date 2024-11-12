import torch.nn as nn

import model.Base.Net as Net

class Model(nn.Module):
    def __init__(self, agent_dim, city_dim, embedding_dim):
        super(Model, self).__init__()
        self.agentEncoder = Net.SelfEncoder([agent_dim], embedding_dim, num_heads=8)
        self.cityEncoder = Net.SelfEncoder([city_dim, city_dim, city_dim], embedding_dim, num_heads=8)
        
