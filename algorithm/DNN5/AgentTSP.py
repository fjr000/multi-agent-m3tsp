import argparse
import time

from model.n4Model.model_v5 import Model
import torch
from utils.TensorTools import _convert_tensor
import numpy as np
from algorithm.DNN5.AgentBase import AgentBase
import tqdm


class Agent(AgentBase):
    def __init__(self, args, config):
        super(Agent, self).__init__(args, config, Model)
        self.model.to(self.device)

    def save_model(self, id):
        filename = f"AgentTSP_{id}"
        super(Agent, self)._save_model(self.args.model_dir, filename)

    def load_model(self, id):
        filename = f"AgentTSP_{id}"
        super(Agent, self)._load_model(self.args.model_dir, filename)