from model.AttnModel.Model import Model, Config
from algorithm.Attn.AgentBase import AgentBase
import argparse


class Agent(AgentBase):
    def __init__(self, args, config):
        super(Agent, self).__init__(args, config, Model)
        self.model.to(self.device)
        self.name = "Attn_AgentV1"

    def save_model(self, id, info = None):
        filename = f"{self.name}_{id}"
        super(Agent, self)._save_model(self.args.model_dir, filename, info)

    def load_model(self, id):
        filename = f"{self.name}_{id}"
        return super(Agent, self)._load_model(self.args.model_dir, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--agent_num", type=int, default=10)
    parser.add_argument("--fixed_agent_num", type=bool, default=False)
    parser.add_argument("--agent_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--grad_max_norm", type=float, default=0.5)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=1e-3)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--city_nums", type=int, default=50)
    parser.add_argument("--random_city_num", type=bool, default=False)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=155008)
    parser.add_argument("--env_masks_mode", type=int, default=7,
                        help="0 for only the min cost  not allow back depot; 1 for only the max cost allow back depot")
    parser.add_argument("--eval_interval", type=int, default=1000, help="eval  interval")
    parser.add_argument("--use_conflict_model", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--train_conflict_model", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--train_actions_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_city_encoder", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--use_agents_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--use_city_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--agents_adv_rate", type=float, default=0.0, help="rate of adv between agents")
    parser.add_argument("--conflict_loss_rate", type=float, default=1.0, help="rate of adv between agents")
    parser.add_argument("--only_one_instance", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--save_model_interval", type=int, default=5000, help="save model interval")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--epoch_size", type=int, default=1280000, help="number of instance for each epoch")
    parser.add_argument("--n_epoch", type=int, default=100, help="number of epoch")
    args = parser.parse_args()

    from envs.GraphGenerator import GraphGenerator as GG

    graphG = GG(args.batch_size, 50, 2)
    graph = graphG.generate()
    from envs.MTSP.MTSP5 import MTSPEnv

    env = MTSPEnv()
    agent = Agent(args, Config)
    min_greedy_cost = 1000
    min_sample_cost = 1000
    loss_list = []

    train_info={
        "use_conflict_model": args.use_conflict_model,
        "train_conflict_model":args.train_conflict_model,
        "train_actions_model": args.train_actions_model,
    }

    city_nums = args.city_nums
    agent_num = args.agent_num

    if args.only_one_instance:
        graph = graphG.generate(1).repeat(args.batch_size, axis=0)
        graph_8 = GG.augment_xy_data_by_8_fold_numpy(graph)
    else:
        graph = graphG.generate(args.batch_size, city_nums)
        graph_8 = GG.augment_xy_data_by_8_fold_numpy(graph)

    output = agent.run_batch_episode(env, graph_8, agent_num, eval_mode=False, info=train_info)
    loss_s = agent.learn(*output)
