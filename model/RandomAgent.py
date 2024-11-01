import numpy as np
from model.AgentBase import AgentBase


def final_action_choice(action_to_chose, action_mask):
    special_action_to_chose = np.where(action_mask == 0)[0] - 1
    return np.concatenate((special_action_to_chose, action_to_chose), axis=-1)


class RandomAgent(AgentBase):
    def __init__(self, config=None):
        super(RandomAgent, self).__init__(config)

    def reset(self, config=None):
        super(RandomAgent, self).reset(config)

    def predict(self, observation, global_mask, agents_action_mask, mode=None):
        action_to_chose = np.where(global_mask == 0)[0] + 1
        actions = np.zeros(self.agent_nums, dtype=np.int32)

        for i in range(self.agent_nums):
            final_action_to_chose = final_action_choice(action_to_chose, agents_action_mask[i])
            actions[i] = np.random.choice(final_action_to_chose)
            if actions[i] != 0 and actions[i] != -1:
                if actions[i] != 1:
                    idx = np.where(action_to_chose == actions[i])[0]
                    action_to_chose = np.delete(action_to_chose, idx)
        return actions
