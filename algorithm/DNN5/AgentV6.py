import torch

from model.n4Model.model_v6 import Model
from algorithm.DNN5.AgentBase import AgentBase


class Agent(AgentBase):
    def __init__(self, args, config):
        super(Agent, self).__init__(args, config, Model)
        self.model.to(self.device)

    def save_model(self, id):
        filename = f"AgentV6_{id}"
        super(Agent, self)._save_model(self.args.model_dir, filename)

    def load_model(self, id):
        filename = f"AgentV6_{id}"
        super(Agent, self)._load_model(self.args.model_dir, filename)

    def _get_action_logprob(self, states, masks, mode="greedy", info=None):
        info = {} if info is None else info
        info.update({
            "mode": mode,
        })
        actions_logits, agents_logits, acts, acts_no_conflict, agents_mask = self.model(states, masks, info)

        # actions_logits = actions_logits / 1.5

        actions_dist = torch.distributions.Categorical(logits=actions_logits)
        act_logp = actions_dist.log_prob(acts)

        if agents_logits is not None:
            agents_dist = torch.distributions.Categorical(logits=agents_logits)
            agents_logp = agents_dist.log_prob(agents_logits.argmax(dim=-1))
            agents_logp = torch.where(agents_mask, agents_logp, 0)
            agt_entropy = torch.where(agents_mask, agents_dist.entropy(), 0)
        else:
            agents_logp = None
            agt_entropy = None

        return acts, acts_no_conflict, act_logp, agents_logp, actions_dist.entropy(), agt_entropy
