class Config(object):
    agent_dim = 13
    embed_dim = 128
    dropout = 0.3

    city_encoder_hidden_dim = 128
    city_encoder_num_layers = 3
    city_encoder_num_heads = 8

    agent_encoder_hidden_dim = 128
    agent_encoder_num_layers = 1
    agent_encoder_num_heads = 8

    action_decoder_hidden_dim = 128
    action_decoder_num_layers = 2
    action_decoder_num_heads = 8

    conflict_deal_hidden_dim = 128
    conflict_deal_num_layers = 2
    conflict_deal_num_heads = 8

    critic_hidden_dim = 256


