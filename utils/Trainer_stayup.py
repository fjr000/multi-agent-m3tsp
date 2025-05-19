
import numpy as np
import random
import torch
import sys

from sympy import floor

sys.path.append("../")
sys.path.append("./")

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
from envs.MTSP.MTSP5 import MTSPEnv
from algorithm.DNN5.AgentV8 import Agent as Agent
import tqdm
from EvalTools import EvalTools
from model.n4Model.config import Config as Config
from envs.GraphGenerator import GraphGenerator as GG


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
def tensorboard_write(writer, train_count, act_loss, agents_loss, stay_up_loss, act_ent_loss, agents_ent_loss, grad, lr):
    writer.add_scalar("train/act_loss", act_loss, train_count)
    writer.add_scalar("train/agents_loss", agents_loss, train_count)
    writer.add_scalar("train/stay_up_loss", stay_up_loss, train_count)
    writer.add_scalar("train/act_ent_loss", act_ent_loss, train_count)
    writer.add_scalar("train/agt_ent_loss", agents_ent_loss, train_count)
    writer.add_scalar("train/grad", grad, train_count)
    writer.add_scalar("train/lr", lr, train_count)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--agent_num", type=int, default=10)
    parser.add_argument("--fixed_agent_num", type=bool, default=False)
    parser.add_argument("--agent_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_max_norm", type=float, default=10)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=3e-3)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--city_nums", type=int, default=50)
    parser.add_argument("--random_city_num", type=bool, default=False)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=6)
    parser.add_argument("--env_masks_mode", type=int, default=7,
                        help="0 for only the min cost  not allow back depot; 1 for only the max cost allow back depot")
    parser.add_argument("--eval_interval", type=int, default=100, help="eval  interval")
    parser.add_argument("--use_conflict_model", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--train_conflict_model", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--train_actions_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_city_encoder", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--use_agents_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--use_city_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--agents_adv_rate", type=float, default=0.0, help="rate of adv between agents")
    parser.add_argument("--conflict_loss_rate", type=float, default=1.0, help="rate of adv between agents")
    parser.add_argument("--only_one_instance", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--save_model_interval", type=int, default=5000, help="save model interval")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
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
    x =  str(args)
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

    for i in tqdm.tqdm(range(100_000_000), mininterval=1):
        # agent_num, city_nums = CC.get_course()

        if args.fixed_agent_num:
            agent_num = args.agent_num
        else:
            # agent_num = np.random.randint(1, args.agent_num + 1)
            def triangular_random(low, high):
                """数值越接近 low，概率越高"""
                return int(np.floor(random.triangular(low, high+1, low)))
            agent_num = triangular_random(low=2, high=args.agent_num)
        if args.random_city_num:
            city_nums = np.random.randint(args.city_nums - 20, args.city_nums+1)
        else:
            city_nums = args.city_nums
        #skip
        # agent_num, city_nums = CC.get_course()

        if args.only_one_instance:
            graph = graphG.generate(1).repeat(args.batch_size, axis=0)
            graph_8 = GG.augment_xy_data_by_8_fold_numpy(graph)
        else:
            graph = graphG.generate(args.batch_size, city_nums)
            graph_8 = GG.augment_xy_data_by_8_fold_numpy(graph)

        output = agent.run_batch_episode(env, graph_8, agent_num, eval_mode=False, info=train_info)
        loss_s = agent.learn(*output)

        tensorboard_write(writer, i,
                          *loss_s,
                          agent.optim.param_groups[0]["lr"]
                          )

        if ((i + 1) % args.eval_interval) == 0:
            if (i+1) % (args.eval_interval * 10) == 0:
                EvalTools.eval_mtsplib(agent, env, writer, i + 1)
            eval_graph = graphG.generate(1, city_nums)
            greedy_gap = EvalTools.eval_random(eval_graph, agent_num, agent, env, writer, i + 1)
            agent.lr_scheduler.step(greedy_gap)
            # last_course = CC.course
            # CC.update_result(greedy_gap / 100)
            # if last_course != CC.course:
            #     print(f"cur course: {CC.course}")

        if (i + 1) % (args.save_model_interval ) == 0:
            agent.save_model(args.agent_id + i + 1)
            print(f"Saving Model {args.agent_id + i + 1}")
