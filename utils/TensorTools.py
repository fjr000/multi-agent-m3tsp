import torch
from typing import Dict, Tuple, Union, List
import numpy as np


def _to_target_shape_dim(data, target_shape_dim=None):
    assert isinstance(target_shape_dim, int) or target_shape_dim is None
    assert isinstance(data, torch.Tensor) or isinstance(data, np.ndarray), "data must be torch.Tensor or np.ndarray"
    res = data
    if target_shape_dim is not None and target_shape_dim != len(data.shape):
        shape_dim = len(res.shape)
        while shape_dim < target_shape_dim:
            res = res.unsqueeze(0)
            shape_dim = len(res.shape)
        while shape_dim > target_shape_dim and res.shape[0] == 1:
            res = res.squeeze(0)
    return res


def _convert_tensor(data, dtype = torch.float32, device="cpu", target_shape_dim=None):
    iter_data = data
    to_iter = False
    if not isinstance(data, Tuple) and not isinstance(data, List):
        iter_data = [data]
        to_iter = True

    for idx, d in enumerate(iter_data):
        if isinstance(d, torch.Tensor):
            d = _to_target_shape_dim(d, target_shape_dim)
            iter_data[idx] = d.to(device)
        else:
            iter_data[idx] = torch.tensor(d, dtype=dtype, device=device)
            iter_data[idx] = _to_target_shape_dim(iter_data[idx], target_shape_dim)

    if to_iter:
        return iter_data[0]
    else:
        return iter_data
