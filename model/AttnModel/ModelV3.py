from model.AttnModel.Model import Model as BaseModel

class Config(object):
    embed_dim = 128
    dropout = 0

    city_encoder_hidden_dim = 128
    city_encoder_num_layers = 3
    city_encoder_num_heads = 8

    action_decoder_hidden_dim = 128
    action_decoder_num_layers = 2
    action_decoder_num_heads = 8

    conflict_deal_hidden_dim = 128
    conflict_deal_num_layers = 1
    conflict_deal_num_heads = 8

    action_hidden_dim = 128
    action_num_layers = 1
    action_num_heads = 8

class Model(BaseModel):
    def __init__(self, config: Config):
        super(Model, self).__init__(Config)

    def forward(self, agent, mask, info=None, eval=False):
        info.update({
            "dones":None
        })
        return super(Model, self).forward(agent, mask, info, eval)
