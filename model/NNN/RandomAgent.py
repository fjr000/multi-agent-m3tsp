import numpy as np
from model.Base.AgentBase import AgentBase

class RandomAgent(AgentBase):
    def __init__(self, config=None):
        super(RandomAgent, self).__init__(config)

    def reset(self, config=None):
        super(RandomAgent, self).reset(config)

    def predict(self, agents_state, global_mask, mode=None):
        action_to_chose = np.where(global_mask == 1)[0] + 1
        actions = np.zeros(self.agent_nums, dtype=np.int32)

        for i in range(self.agent_nums):
            if len(action_to_chose) == 0:
                actions[i]=0
                continue
            actions[i] = np.random.choice(action_to_chose)
            if actions[i] != 0 and actions[i] != -1:
                if actions[i] != 1:
                    idx = np.where(action_to_chose == actions[i])[0]
                    action_to_chose = np.delete(action_to_chose, idx)
        return actions
