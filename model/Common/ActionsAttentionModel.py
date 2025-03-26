import torch
import torch.nn as nn
from model.Common.CityAttentionEncoder import CityAttentionEncoder
from model.Common.AgentAttentionEncoder import AgentAttentionEncoder, AgentAttentionRNNEncoder
from model.Common.AgentCityAttentionDecoder import AgentCityAttentionDecoder
from model.Common.config import ActionsModelConfig

class ActionsAttentionModel(nn.Module):
    def __init__(self, config: ActionsModelConfig):
        super(ActionsAttentionModel, self).__init__()

        self.agent_encoder = AgentAttentionEncoder(input_dim=2,
                                                   hidden_dim=config.agent_encoder_hidden_dim,
                                                   embed_dim=config.embed_dim,
                                                   num_heads=config.agent_encoder_num_heads,
                                                   num_layers=config.agent_encoder_num_layers,
                                                   norm="layer")

        self.agent_city_decoder = AgentCityAttentionDecoder(hidden_dim=config.action_decoder_hidden_dim,
                                                            embed_dim=config.embed_dim,
                                                            num_heads=config.action_decoder_num_heads,
                                                            num_layers=config.action_decoder_num_layers,
                                                            norm="layer")

        self.city_embed = None
        self.city_embed_mean = None

    def init_rnn_state(self, batch_size, agent_num,device):
        if isinstance(self.agent_encoder, AgentAttentionEncoder):
            pass
        else:
            self.agent_encoder.init_rnn_state(batch_size, agent_num, device)

    def forward(self, city_embed, city_embed_mean, agent, agent_mask = None, agent_city_mask = None):

        n_agents = agent.size(1)
        agent_self_embed = self.agent_encoder(city_embed[:,:-n_agents,:], city_embed[:,-n_agents:,:], city_embed_mean, agent, agent_mask)

        agent_city_embed, act_logits = self.agent_city_decoder(agent_self_embed, city_embed, masks = agent_city_mask)

        return agent_city_embed, act_logits

if __name__ == '__main__':
    encoder = CityAttentionEncoder(4, 128, 2, 2, 'batch', 256)
    city = torch.randn((2, 10, 2))
    city_mask = torch.randint(0, 2, (2, 10, 10)).bool()
    city_embed, city_embed_mean = encoder(city, city_mask)

    agent_encoder =AgentAttentionEncoder(input_dim=2, hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2, norm='batch')
    agent1 = torch.randint(0,10,(2,3,2))
    agent2 = torch.randn((2,3,8))
    agents_mask = torch.randint(0, 2, (2, 3, 3)).bool()
    agent = torch.cat([agent1, agent2], dim=-1)
    agents_embed = agent_encoder(city_embed, city_embed_mean, agent, agents_mask)

    agent_city_decoer = AgentCityAttentionDecoder(hidden_dim=256, embed_dim=128, num_heads=4, num_layers=2, norm = "batch")
    agent_city_mask = torch.randint(0, 2, (2, 3, 10)).bool()
    o = agent_city_decoer(agents_embed, city_embed, agent_city_mask)

    action_model = ActionsAttentionModel(config=ActionsModelConfig)
    action_model.init_city(city, city_mask)
    agent_city_embed, act_logits = action_model(agent, agents_mask, agent_city_mask)
