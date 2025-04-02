import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=10000):
        super().__init__()
        self.d_model = d_model

        self.encoding = torch.zeros(max_seq_len, d_model)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, dtype=torch.float32)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, seq_len):
        return self.encoding[:seq_len, :]
