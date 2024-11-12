
class AgentBase:
    def __init__(self, config):
        self.config = config
        self.city_nums = None
        self.agent_nums = 1
        self.parse_config(config)

    def reset(self, config):
        self.config = config
        self.parse_config(config)

    def parse_config(self, config):
        self.config = config
        if config is not None:
            city_nums = config.get("city_nums")
            if city_nums is not None:
                self.city_nums = city_nums

            agent_nums = config.get("agent_nums")
            if agent_nums is not None:
                self.agent_nums = agent_nums

    def forward(self, agent_state, city_state, index):
        raise NotImplementedError

    def predict(self, agents_state, cities_state, global_mask, agents_action_mask, mode=None):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError
