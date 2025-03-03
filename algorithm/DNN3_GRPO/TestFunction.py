import torch
from torch.distributions.categorical import Categorical
import numpy as np
if __name__ == '__main__':
    k = 7
    ipt = torch.randn((1,1,8))
    mask = torch.randint(0,2,(1,1,8))
    masked_ipt = ipt.masked_fill(mask < 1e-8, -10e8)
    probs = Categorical(logits=masked_ipt)

    p, act = torch.topk(probs.probs,k,dim = -1)
    masked_act = act[p>0].cpu().numpy()
    actions = np.concatenate((masked_act[...,np.newaxis]//64, masked_act[...,np.newaxis]%64), axis = -1)
    actions.tolist()


def __get_action_topk(self, feature, masks, k = 4):
    grid, blocks, extra = feature[0], feature[1], feature[-1]
    act_logits = self.actor(grid.unsqueeze(0), blocks.unsqueeze(0), extra.unsqueeze(0))
    masked_logits = act_logits.masked_fill(masks.unsqueeze(0).unsqueeze(0) < 1e-8, -10e9)
    probs = Categorical(logits=masked_logits)

    p, act = torch.topk(probs.probs,k,dim = -1)
    masked_act = act[p>0].cpu().numpy()
    actions = np.concatenate((masked_act[...,np.newaxis]//64, masked_act[...,np.newaxis]%64), axis = -1)
    return actions.tolist()