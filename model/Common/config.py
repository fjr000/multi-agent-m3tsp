
class CityEncoderConfig:
    embed_dim = 128

    city_encoder_hidden_dim = 128
    city_encoder_num_layers = 3
    city_encoder_num_heads = 8

class ActionsModelConfig(object):
    embed_dim = 128
    dropout = 0.3

    agent_encoder_hidden_dim = 128
    agent_encoder_num_layers = 2
    agent_encoder_num_heads = 8

    action_decoder_hidden_dim = 128
    action_decoder_num_layers = 1
    action_decoder_num_heads = 8


class ConflictModelConfig:
    embed_dim = 128

    agent_encoder_hidden_dim = 128
    agent_encoder_num_layers = 1
    agent_encoder_num_heads = 8

    conflict_deal_hidden_dim = 128
    conflict_deal_num_layers = 1
    conflict_deal_num_heads = 8

class ModelConfig:
    city_encoder_config = CityEncoderConfig()
    actions_model_config = ActionsModelConfig()
    conflict_model_config = ConflictModelConfig()