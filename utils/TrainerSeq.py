
import numpy as np
import random
import torch
import sys

sys.path.append("../")
sys.path.append("./")

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
from envs.MTSP.MTSP5_IDVR import MTSPEnv_IDVR as MTSPEnv
from algorithm.SeqAgent.AgentV1 import AgentV1 as Agent
import tqdm
from EvalTools import EvalTools
from model.SeqModel.config import ModelConfig as Config
from envs.GraphGenerator import GraphGenerator as GG
from utils.TensorTools import _convert_tensor
from torch.utils.data import RandomSampler

class BufferSampler:
    def __init__(self, sample_size):
        self.sample_size = sample_size
        self.graph = []
        self.state = []
        self.act_logp = []
        self.act = []
        self.act_mask = []
        self.gae = []
        self.returns = []
        self.V = []
        self.size = 0
        self.reset()
    def reset(self):
        del self.graph, self.state, self.act_logp, self.act, self.act_mask, self.gae, self.returns, self.V
        self.graph = []
        self.state = []
        self.act_logp = []
        self.act = []
        self.act_mask = []
        self.gae = []
        self.returns = []
        self.V = []
        self.size = 0

    def insert(self,batch_graph, states, act_logp, act, act_mask, gae, returns, V):
        mask = np.any(~np.isclose(act_logp, np.zeros_like(act_logp), 1e-8, 1e-10), axis=1)
        self.graph.append(batch_graph[mask])
        self.state.append(states[mask])
        self.act_logp.append(act_logp[mask])
        self.act.append(act[mask])
        self.act_mask.append(act_mask[mask])
        self.gae.append(gae[mask])
        self.returns.append(returns[mask])
        self.V.append(V[mask])
        self.size += self.V[-1].shape[0]

    def ready(self, device):
        self.graph = _convert_tensor(np.concatenate(self.graph, axis=0), dtype=torch.float32, device=device)
        self.state = _convert_tensor(np.concatenate(self.state, axis=0), dtype=torch.float32, device=device)
        self.act_logp = _convert_tensor(np.concatenate(self.act_logp, axis=0), dtype=torch.float32, device=device)
        self.act = _convert_tensor(np.concatenate(self.act, axis=0), dtype=torch.int, device=device)
        self.act_mask = _convert_tensor(np.concatenate(self.act_mask, axis=0), dtype=torch.bool, device=device)
        self.gae = _convert_tensor(np.concatenate(self.gae, axis=0), dtype=torch.float32, device=device)
        self.returns = _convert_tensor(np.concatenate(self.returns, axis=0), dtype=torch.float32, device=device)
        self.V = _convert_tensor(np.concatenate(self.V, axis=0), dtype=torch.float32, device=device)

    def sample(self, sample_size):
        assert isinstance(self.V,torch.Tensor)
        # 创建随机采样器（不放回采样）
        B= self.V.size(0)
        indices = torch.randperm(B)[:sample_size]  # 生成随机索引
        return (
                self.graph[indices].detach(),
                self.state[indices].detach(),
                self.act_logp[indices].detach(),
                self.act[indices].detach(),
                self.act_mask[indices].detach(),
                self.gae[indices].detach(),
                self.returns[indices].detach(),
                self.V[indices].detach(),
                )



def set_seed(seed=42):
    # 基础库
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch核心设置
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU时

    # # 禁用CUDA不确定算法
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def tensorboard_write(writer, train_count, policy_loss, value_loss, entropy_loss, lr):
    writer.add_scalar("train/policy_loss", policy_loss, train_count)
    writer.add_scalar("train/value_loss", value_loss, train_count)
    writer.add_scalar("train/entropy_loss", entropy_loss, train_count)
    writer.add_scalar("train/lr", lr, train_count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--agent_num", type=int, default=1)
    parser.add_argument("--fixed_agent_num", type=bool, default=True)
    parser.add_argument("--agent_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_max_norm", type=float, default=1)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=1e-2)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--city_nums", type=int, default=50)
    parser.add_argument("--random_city_num", type=bool, default=False)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=0)
    parser.add_argument("--env_masks_mode", type=int, default=4,
                        help="0 for only the min cost  not allow back depot; 1 for only the max cost allow back depot")
    parser.add_argument("--eval_interval", type=int, default=1, help="eval  interval")
    parser.add_argument("--use_conflict_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_conflict_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_actions_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_city_encoder", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--use_agents_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--use_city_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--agents_adv_rate", type=float, default=0.5, help="rate of adv between agents")
    parser.add_argument("--conflict_loss_rate", type=float, default=0.1, help="rate of adv between agents")
    parser.add_argument("--only_one_instance", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--save_model_interval", type=int, default=10000, help="save model interval")
    parser.add_argument("--seed", type=int, default=528, help="random seed")
    parser.add_argument("--buffer_min_size", type=int, default=4096 , help="if buffer size > buffer_min_size train begin")
    parser.add_argument("--sample_size", type=int, default=1024, help="train size")
    parser.add_argument("--K_epoch", type=int, default=4, help="use the old traj train times")
    args = parser.parse_args()

    set_seed(args.seed)

    fig = None
    graphG = GG(args.batch_size, args.city_nums, 2)
    env = MTSPEnv({
        "env_masks_mode": args.env_masks_mode,
        "use_conflict_model":args.use_conflict_model
    })
    from algorithm.OR_Tools.mtsp import ortools_solve_mtsp

    # indexs, cost, used_time = ortools_solve_mtsp(graph, args.agent_num, 10000)
    # env.draw(graph, cost, indexs, used_time, agent_name="or_tools")
    # print(f"or tools :{cost}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f"../log/workflow-{timestamp}")
    writer.add_text("agent_config", str(args), 0)
    x = str(args)
    agent = Agent(args, Config)
    agent.load_model(args.agent_id)
    for lr in agent.optim.param_groups:
        lr["lr"] = args.lr
    from CourseController import CourseController

    CC = CourseController()
    agent_num, city_nums = args.agent_num, args.city_nums

    train_info={
        "use_conflict_model": args.use_conflict_model,
        "train_conflict_model":args.train_conflict_model,
        "train_actions_model": args.train_actions_model,
    }

    buf = BufferSampler(args.sample_size)

    for i in tqdm.tqdm(range(100_000_000), mininterval=1):
        # agent_num, city_nums = CC.get_course()

        if args.fixed_agent_num:
            agent_num = args.agent_num
        else:
            agent_num = np.random.randint(1, args.agent_num + 1)

        if args.random_city_num:
            city_nums = np.random.randint(agent_num*5, args.city_nums+1)
        else:
            city_nums = args.city_nums

        if args.only_one_instance:
            graph = graphG.generate(1).repeat(args.batch_size, axis=0)
            graph_8 = GG.augment_xy_data_by_8_fold_numpy(graph)
        else:
            graph = graphG.generate(args.batch_size, city_nums)
            graph_8 = GG.augment_xy_data_by_8_fold_numpy(graph)
        # x = []
        # for _ in range(4):
        #     output = agent.run_batch_episode(env, graph_8, agent_num, eval_mode=False)
        #     x.append(output)
        # loss = [0,0,0]
        # for _ in range(3):
        #     for t in range(4):
        #         loss_s = agent.learn(*x[t])
        #         loss[0] += loss_s[0] / 3/4
        #         loss[1] += loss_s[1] / 3/4
        #         loss[2] += loss_s[2] / 3/4
        buf.reset()
        len = 0

        while len < args.buffer_min_size:
            output = agent.run_batch_episode(env, graph_8, agent_num, eval_mode=False)
            buf.insert(*output)
            len += buf.size
        buf.ready(agent.device)

        loss = [0,0,0]
        for _ in range(args.K_epoch):
            loss_ = agent.learn(*buf.sample(args.sample_size))
            loss[0] += loss_[0] / args.K_epoch
            loss[1] += loss_[1] / args.K_epoch
            loss[2] += loss_[2] / args.K_epoch

        tensorboard_write(writer, i,
                          *loss,
                          agent.optim.param_groups[0]["lr"]
                          )

        if ((i + 1) % args.eval_interval) == 0:
            EvalTools.eval_mtsplib(agent, env, writer, i + 1)
            eval_graph = graphG.generate(1, city_nums)
            greedy_gap = EvalTools.eval_random(eval_graph, agent_num, agent, env, writer, i + 1, True)
            agent.lr_scheduler.step(greedy_gap)

        if (i + 1) % (args.save_model_interval ) == 0:
            agent.save_model(args.agent_id + i + 1)
