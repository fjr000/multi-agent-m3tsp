import copy

from envs.MTSP.MTSP4 import MTSPEnv
import numpy as np
class MTSPEnv_IDVR(MTSPEnv):
    def __init__(self, config):
        super(MTSPEnv_IDVR, self).__init__(config)
        self.last_costs = None

    def _init(self, graph=None):
        super(MTSPEnv_IDVR)._init(graph)
        self.last_costs = copy.deepcopy(self.costs)


    def _get_reward(self):
        def _get_reward(self):
            self.dones = np.all(self.stage_2, axis=1)
            self.done = np.all(self.dones, axis=0)
            # self.rewards = np.where(self.dones[:,None], -np.max(self.costs,keepdims=True, axis=1).repeat(self.salesmen, axis = 1), 0)
            self.rewards = self.last_costs - self.costs
            self.last_costs = self.costs
            return self.rewards

