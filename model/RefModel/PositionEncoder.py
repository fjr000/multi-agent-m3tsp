import torch
import torch.nn as nn
import math


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=1000):
        super().__init__()
        self.d_model = d_model

        # 创建位置编码
        position_encoding = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('position_encoding', position_encoding)
        self.requires_grad_(False)

    def forward(self, seq_len):
        # positions: [batch_size, seq_len]
        return self.position_encoding[:seq_len, :]
