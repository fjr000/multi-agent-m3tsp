from model.ET.model_v1 import Model

from algorithm.ET.AgentBase import AgentBase


class AgentV1(AgentBase):
    def __init__(self, args, config):
        super(AgentV1, self).__init__(args, config, Model)
        self.model.to(self.device)

    def save_model(self, id):
        filename = f"ETAgentV1_{id}"
        super(AgentV1, self)._save_model(self.args.model_dir, filename)

    def load_model(self, id):
        filename = f"ETAgentV1_{id}"
        super(AgentV1, self)._load_model(self.args.model_dir, filename)
