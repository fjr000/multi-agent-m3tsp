import collections

import numpy as np
import random
import torch
import sys
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
import tqdm

sys.path.append("../")
sys.path.append("./")

from envs.MTSP.MTSP5 import MTSPEnv
from algorithm.Attn.AgentV2 import Agent as Agent
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
def tensorboard_write(writer, train_count, learn_info, lr):
    for k, v in learn_info.items():
        if v is None:
            continue

        writer.add_scalar(f"train/{k}", v, global_step=train_count)
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
    parser.add_argument("--grad_max_norm", type=float, default=0.5)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=1e-3)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--city_nums", type=int, default=50)
    parser.add_argument("--random_city_num", type=bool, default=False)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=4)
    parser.add_argument("--env_masks_mode", type=int, default=7,
                        help="0 for only the min cost  not allow back depot; 1 for only the max cost allow back depot")
    parser.add_argument("--eval_interval", type=int, default=100, help="eval  interval")
    parser.add_argument("--use_conflict_model", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--train_conflict_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_actions_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_city_encoder", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--use_agents_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--use_city_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--agents_adv_rate", type=float, default=0.0, help="rate of adv between agents")
    parser.add_argument("--conflict_loss_rate", type=float, default=1.0, help="rate of adv between agents")
    parser.add_argument("--only_one_instance", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--save_model_interval", type=int, default=200, help="save model interval")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--epoch_size", type=int, default=1280000, help="number of instance for each epoch")
    parser.add_argument("--n_epoch", type=int, default=100, help="number of epoch")
    args = parser.parse_args()

    random.seed(528)
    np.random.seed(528)
    eval_agent_num = 3
    eval_city_num = 50
    eval_batch_size = 100
    eval_min_cost = 10086
    eval_graphG = GG(eval_batch_size, eval_city_num, 2)
    eval_graph = eval_graphG.generate()

    set_seed(args.seed)

    fig = None
    graphG = GG(args.batch_size, args.city_nums, 2)
    env = MTSPEnv({
        "env_masks_mode": args.env_masks_mode,
        "use_conflict_model":args.use_conflict_model
    })

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f"../log/workflow-{timestamp}")
    writer.add_text("agent_config", str(args), 0)
    x =  str(args)

    agent = Agent(args, Config)
    info = agent.load_model(args.agent_id)

    agent_num, city_nums = args.agent_num, args.city_nums

    train_info={
        "use_conflict_model": args.use_conflict_model,
        "train_conflict_model":args.train_conflict_model,
        "train_actions_model": args.train_actions_model,
    }

    greedy_cost, greedy_traj, greedy_time = EvalTools.EvalGreedy(eval_graph, eval_agent_num, agent, env)
    writer.add_scalar("eval/cost", greedy_cost, 0)
    print(f"agent_num:{eval_agent_num},city_num:{eval_graph.shape[1]},"
          f"costs:{greedy_cost:.5f},"
          )
    eval_min_cost = min(greedy_cost, eval_min_cost)

    step_epoch = args.epoch_size // args.batch_size

    total_step = 0 if info is None else info.get("total_step", 0)
    start_epoch = total_step // step_epoch
    start_step = total_step % step_epoch

    save_info = {
        "args": args,
        "total_step": total_step,
    }

    buffer_list = collections.deque(maxlen=10)

    for epoch in range(start_epoch, args.n_epoch):
        print("start epoch:",epoch)
        _start_step = start_step
        if epoch == start_epoch:
            _start_step = start_step
        else:
            _start_step = 0

        for i in tqdm.tqdm(range(_start_step, step_epoch), mininterval=1):
            # agent_num, city_nums = CC.get_course()
            total_step = epoch * step_epoch + i +1
            save_info.update({"total_step": total_step})

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

            if args.only_one_instance:
                graph = graphG.generate(1).repeat(args.batch_size, axis=0)
                graph_8 = GG.augment_xy_data_by_8_fold_numpy(graph)
            else:
                graph = graphG.generate(args.batch_size, city_nums)
                graph_8 = GG.augment_xy_data_by_8_fold_numpy(graph)

            with torch.no_grad():
                output = agent.run_batch_episode(env, graph_8, agent_num, eval_mode=False, info=train_info)
            buffer_list.append(output)
            # for i in range(3):
            loss_s = agent.learn(buffer_list[-1])

            tensorboard_write(writer, i,
                              loss_s,
                              agent.optim.param_groups[0]["lr"]
                              )

            if total_step % (args.accumulation_steps * args.eval_interval) == 0:
                greedy_cost, greedy_traj, greedy_time = EvalTools.EvalGreedy(eval_graph, eval_agent_num, agent, env)
                writer.add_scalar("eval/cost", greedy_cost, total_step)
                print(f"total_step:{total_step}, agent_num:{eval_agent_num},city_num:{eval_graph.shape[1]},"
                      f"costs:{greedy_cost:.5f},"
                      )
                agent.lr_scheduler.step(greedy_cost)
                if greedy_cost < eval_min_cost:
                    agent.save_model(999_999_999, save_info)
                eval_min_cost = min(greedy_cost, eval_min_cost)

            if total_step % (args.accumulation_steps *  args.save_model_interval) == 0:
                agent.save_model(epoch + 1, save_info)

        agent.save_model(epoch + 1, save_info)
