import torch

from model.n4Model.model_v8 import Model
from algorithm.DNN5.AgentBase import AgentBase


class Agent(AgentBase):
    def __init__(self, args, config):
        super(Agent, self).__init__(args, config, Model)
        self.model.to(self.device)

    def save_model(self, id):
        filename = f"AgentV8_{id}"
        super(Agent, self)._save_model(self.args.model_dir, filename)

    def load_model(self, id):
        filename = f"AgentV8_{id}"
        super(Agent, self)._load_model(self.args.model_dir, filename)
