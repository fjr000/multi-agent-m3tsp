from model.n4Model.model_v4 import Model as BaseModel
import torch
import torch.nn as nn
from model.n4Model.config import Config


class CriticModel(nn.Module):
    def __init__(self, embed_dim):
        super(CriticModel, self).__init__()
        self.embed_dim = embed_dim
        self.V = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim // 2, 1),
        )

    def forward(self, x):
        return self.V(x)


class Model(BaseModel):
    def __init__(self, config: Config):
        super(Model, self).__init__(config)
        self.critic = CriticModel(config.embed_dim)

    def forward(self, agent, mask, info=None):
        out = super(Model, self).forward(agent, mask, info)
        V = self.critic(self.actions_model.agent_decoder.agent_embed)
        return *out, V
