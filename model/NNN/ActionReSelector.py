import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, List
import copy


class ActionReSelectorBySeq:
    def __init__(self, action_num =None, action_range: Tuple = None):
        self.action_range = action_range
        self.action_num = action_num

    def reSelect(self, actions: List):
        if self.action_num is not None:
            assert len(actions) == self.action_num, f"the length of actions must be equal to init_action_num:{self.action_num}"
        if self.action_range is not None:
            assert (min(actions) >= self.action_range[0] and max(actions) <= self.action_range[1]), f"actions must between:{self.action_range[0]} and {self.action_range[1]}"
        action_dict = {}
        return_actions = copy.deepcopy(actions)
        for idx, act in enumerate(actions):
            if act == -1 or act == 0:
                continue
            else:
                if act not in action_dict.keys():
                    action_dict[act] = idx
                else:
                    act = 0
            return_actions[idx] = act
        return return_actions


if __name__ == '__main__':
    ARSBS = ActionReSelectorBySeq(action_range=(-1, 5))
    print(ARSBS.reSelect(actions=[-1, 0,5,1,4,5,3,2,1,5,4]))
