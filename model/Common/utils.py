import torch.nn as nn


def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.GRU):
            for param in layer.parameters():
                if param.dim() > 1:  # 仅对 weight 参数初始化
                    nn.init.orthogonal_(param)
