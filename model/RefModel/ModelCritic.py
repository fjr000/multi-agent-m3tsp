import torch
import torch.nn as nn
from model.RefModel.CityAttentionEncoder import CityAttentionEncoder
from model.RefModel.ActionsAttentionModel import ActionsAttentionModel
from model.RefModel.ConflictAttentionModel import ConflictAttentionModel
from model.RefModel.AgentAttentionCritic import AgentAttentionCritic
from model.RefModel.config import ModelConfig

class Model(nn.Module):
    def __init__(self, config:ModelConfig, args = None):
        super(Model, self).__init__()
        self.config = config
        self.args = args
        self.city_encoder = CityAttentionEncoder(config.city_encoder_config.city_encoder_num_heads,
                                                 config.city_encoder_config.embed_dim,
                                                 config.city_encoder_config.city_encoder_num_layers,
                                                 2,
                                                 'batch',
                                                 config.city_encoder_config.city_encoder_hidden_dim,)
        self.actions_model = ActionsAttentionModel(config.actions_model_config)
        self.conflict_model = ConflictAttentionModel(config.conflict_model_config)
        self.critic = AgentAttentionCritic(config.actions_model_config.embed_dim, config.actions_model_config.embed_dim)
        self.step = 0
        self.city_embed = None
        self.city_embed_mean = None

    def init_city(self, city, mask = None):
        if self.args.train_city_encoder:
            self.city_embed, self.city_embed_mean = self.city_encoder(city, mask)
        else:
            with torch.no_grad():
                self.city_embed, self.city_embed_mean = self.city_encoder(city, mask)
        if mask is None:
            self.step = 0

    def forward(self, agent, mask, info = None):
        mode = "greedy" if info is None else info.get("mode", "greedy")
        use_conflict_model = True if info is None else info.get("use_conflict_model", True)
        agent_mask = None if info is None else info.get("masks_in_salesmen", None)

        # if self.step == 0:
        #     self.actions_model.init_rnn_state(agent.size(0),agent.size(1),agent.device)

        if self.args.use_city_mask:
            city_mask = None if info is None else info.get("mask", None)
            assert city_mask is not None, "use_city_mask requires city mask"
            self.init_city(agent, city_mask)

        if self.args.train_actions_model:
            agents_embed, actions_logits = self.actions_model(self.city_embed, self.city_embed_mean,agent, agent_mask, mask)
        else:
            with torch.no_grad():
                agents_embed, actions_logits = self.actions_model(self.city_embed, self.city_embed_mean,agent, agent_mask, mask)

        acts = None
        if mode == "greedy":
            # 1. 获取初始选择 --------------------------------------------------------
            acts = actions_logits.argmax(dim=-1)
            # if self.step == 0:
            # acts_p = nn.functional.softmax(actions_logits, dim=-1)
            #     _, acts  = acts_p[:,0,:].topk(agent.size(1), dim=-1)
        elif mode == "sample":
            acts = torch.distributions.Categorical(logits=actions_logits).sample()
        else:
            raise NotImplementedError
        if use_conflict_model:
            if self.args.train_conflict_model:
                agents_logits = self.conflict_model(agents_embed, self.city_embed, acts)
            else:
                with torch.no_grad():
                    agents_logits = self.conflict_model(agents_embed, self.city_embed, acts)

            agents = agents_logits.argmax(dim=-1)

            # pos = torch.arange(agents_embed.size(1), device=agents.device).unsqueeze(0).expand(agent.size(0), -1)
            pos = torch.arange(agents_embed.size(1), device=agents.device).unsqueeze(0)
            masks = torch.logical_or(agents == pos, acts == 0)
            del pos
            acts_no_conflict = torch.where(masks, acts, -1)
        else:
            agents_logits = None
            acts_no_conflict = acts
            masks = None
        self.step += 1
        V = self.critic(agents_embed)
        # a = torch.all((acts_no_conflict == -1),dim = -1)
        # if torch.any(a):
        #     pass
        return actions_logits, agents_logits, acts, acts_no_conflict, masks, V

