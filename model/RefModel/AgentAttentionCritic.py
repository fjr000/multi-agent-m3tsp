import torch
import torch.nn as nn
class AgentAttentionCritic(nn.Module):
    def __init__(self, embed_dim, hidden_size):
        super(AgentAttentionCritic, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

    def forward(self, agent_embed, mode = "None"):
        v = self.layer(agent_embed)
        if mode == 'mean':
            v = v.mean(dim=1)
        elif mode == 'max':
            v = v.max(dim=1)[0]
        elif mode == 'sum':
            v = v.sum(dim=1)

        return v

