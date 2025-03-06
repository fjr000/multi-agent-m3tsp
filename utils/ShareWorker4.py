import sys
import time

import numpy as np
import sys

sys.path.append("../")
sys.path.append("./")

import torch.multiprocessing as mp
from envs.GraphGenerator import GraphGenerator as GG
from utils.TensorTools import _convert_tensor
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from algorithm.OR_Tools.mtsp import ortools_solve_mtsp
import argparse
from envs.MTSP.MTSP3 import MTSPEnv
from algorithm.DNN4.AgentV2 import AgentV2 as Agent
import tqdm
from CourseController import CourseController

torch.set_num_threads(1)


def worker_process(share_agent, agent_class, args, env_class, env_config, recv_pipe, queue):
    env = env_class(env_config)
    work_agent = share_agent
    # work_agent = agent_class(args)
    while True:
        graph, agent_num = recv_pipe.recv()
        # work_agent.model.load_state_dict(share_agent.model.state_dict())
        features_nb, actions_nb, actions_no_conflict_nb, returns_nb, individual_returns_nb, masks_nb, dones_nb = work_agent.run_batch(env, graph, agent_num,
                                                                              args.batch_size // args.num_worker)
        queue.put((graph, features_nb, actions_nb, actions_no_conflict_nb, returns_nb,individual_returns_nb, masks_nb, dones_nb))


def eval_process(share_agent, agent_class, args, env_class, env_config, recv_model_pipe, send_result_pipe, sample_times):
    env = env_class(env_config)
    eval_agent = share_agent
    # eval_agent = agent_class(args)
    print(eval_agent.device)
    while True:
        graph, agent_num = recv_model_pipe.recv()
        # eval_agent.model.load_state_dict(share_agent.model.state_dict())
        st = time.time_ns()
        greedy_cost, greedy_trajectory = eval_agent.eval_episode(env, graph, agent_num, exploit_mode="greedy")
        ed = time.time_ns()
        greedy_time = (ed - st) / 1e9
        min_sample_cost = np.inf
        min_sample_trajectory = None
        st = time.time_ns()
        # for i in range(sample_times):
        #     sample_cost, sample_trajectory = eval_agent.eval_episode(env, graph, agent_num, exploit_mode="sample")
        #     if sample_cost < min_sample_cost:
        #         min_sample_cost = sample_cost
        #         min_sample_trajectory = sample_trajectory
        ed = time.time_ns()
        sample_time = (ed - st) / 1e9

        ortools_trajectory, ortools_cost, used_time = ortools_solve_mtsp(graph, agent_num, 10000)
        # env.draw_multi(graph,
        #                [ortools_cost, greedy_cost, min_sample_cost],
        #                [ortools_trajectory, greedy_trajectory, min_sample_trajectory],
        #                [used_time, greedy_time, sample_time],
        #                ["or_tools", "greedy", "sample"]
        #                )

        send_result_pipe.send(
            (greedy_cost, greedy_trajectory, min_sample_cost, min_sample_trajectory, ortools_cost, ortools_trajectory))


def train_process(share_agent, agent_class, agent_args, send_pipes, queue, eval_model_pipe, eval_result_pipe):
    # agent_args.use_gpu = False

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f"../log/workflow-{timestamp}")
    writer.add_text("agent_config", str(agent_args), 0)

    eval_count = 0
    train_count = 0
    graphG = GG(1, agent_args.city_nums)

    from model.n4Model.config import Config
    train_agent = agent_class(agent_args, Config)
    model_state_dict = share_agent.model.state_dict()
    train_agent.model.load_state_dict(model_state_dict)
    CC = CourseController()
    for _ in tqdm.tqdm(range(100_000_000)):
        # cur_city_nums = np.random.randint(agent_args.city_nums, agent_args.max_city_nums+1)
        # cur_agents_num = np.random.randint(agent_args.agent_num, agent_args.max_agent_num+1)
        cur_agents_num, cur_city_nums = CC.get_course()
        graph = graphG.generate(num = cur_city_nums)
        # if (train_count+1) % agent_args.num_worker == 0:
        for pipe in send_pipes:
            pipe.send((graph, cur_agents_num))

        state_nb_list = []
        actions_nb_list = []
        actions_no_conflict_nb_list = []
        returns_nb_list = []
        individual_returns_nb_list = []
        masks_nb_list = []
        dones_nb_list = []

        for i in range(agent_args.num_worker):
            graph, features_nb, actions_nb,actions_no_conflict_nb, returns_nb, individual_returns_nb, masks_nb, dones_nb = queue.get()
            state_nb_list.append(features_nb)
            actions_nb_list.append(actions_nb)
            actions_no_conflict_nb_list.append(actions_no_conflict_nb)
            returns_nb_list.append(returns_nb)
            individual_returns_nb_list.append(individual_returns_nb)
            # reward_nb_list.append(reward_nb)
            masks_nb_list.append(masks_nb)
            dones_nb_list.append(dones_nb)
            # logp_nb_list.append(logp_nb)
        features_nb = np.concatenate(state_nb_list, axis=0)
        actions_nb = np.concatenate(actions_nb_list, axis=0)
        actions_no_conflict_nb = np.concatenate(actions_no_conflict_nb_list, axis=0)
        returns_nb = np.concatenate(returns_nb_list, axis=0)
        individual_returns_nb = np.concatenate(individual_returns_nb_list, axis=0)
        # rewards_nb = np.concatenate(reward_nb_list, axis=0)
        masks_nb = np.concatenate(masks_nb_list, axis=0)
        dones_nb = np.concatenate(dones_nb_list,axis = 0)
        # logp_nb = np.concatenate(logp_nb_list,axis = 0)
        train_agent.model.load_state_dict(share_agent.model.state_dict())
        graph = graph - graph[0,0]
        loss = train_agent.learn(_convert_tensor(graph, dtype=torch.float32, device=train_agent.device, target_shape_dim=3),
                           _convert_tensor(features_nb, dtype=torch.float32, device=train_agent.device),
                           _convert_tensor(actions_nb, dtype=torch.float32, device=train_agent.device),
                           # _convert_tensor(returns_nb, dtype=torch.float32, device=train_agent.device),
                             _convert_tensor(individual_returns_nb, dtype=torch.float32, device=train_agent.device),
                            # rewards_nb,
                           _convert_tensor(masks_nb, dtype=torch.float32, device=train_agent.device),
                            dones_nb,
                            # _convert_tensor(logp_nb, dtype=torch.float32, device=train_agent.device),
                           )
        writer.add_scalar("loss", loss, train_count)
        torch.cuda.empty_cache()  # 清理未使用的缓存
        share_agent.model.load_state_dict(train_agent.model.state_dict())

        if (train_count + 1) % 1 == 0:
            eval_count = train_count
            # graph = graphG.generate()
            eval_model_pipe.send((graph, cur_agents_num))

        train_count += 1

        if (train_count +1)% 1 == 0 and eval_result_pipe.poll():
            greedy_cost, greedy_trajectory, min_sample_cost, min_sample_trajectory, ortools_cost, ortools_trajectory = eval_result_pipe.recv()
            writer.add_scalar("greedy_cost", greedy_cost, eval_count)
            writer.add_scalar("min_sample_cost", min_sample_cost, eval_count)
            writer.add_scalar("ortools_cost", ortools_cost, eval_count)
            greedy_gap = (greedy_cost - ortools_cost) / ortools_cost * 100
            sample_gap = (min_sample_cost - ortools_cost) / ortools_cost * 100
            writer.add_scalar("greedy_gap", greedy_gap, eval_count)
            writer.add_scalar("sample_gap", sample_gap, eval_count)

            print(f"greddy_cost:{greedy_cost},{greedy_gap}%, sample_cost:{min_sample_cost},{sample_gap}%, ortools_cost:{ortools_cost}")
            last_course = CC.course
            CC.update_result(greedy_gap/100)
            if CC.course != last_course:
                print(f"next course: {CC.course}")

        if (train_count + 1) % 5000 == 0:
            train_agent.save_model(train_count + 1 + agent_args.agent_id)


class SharelWorker:
    def __init__(self, agent_class, args, env_class):
        self.agent_class = agent_class
        self.agent_args = args
        self.env_class = env_class
        self.env_config = {
            "salesmen": args.agent_num,
            "cities": args.city_nums,
            "seed": None,
            "mode": 'rand'
        }
        self.num_worker = args.num_worker

        self.queue = mp.Queue()
        self.worker_pipes = [mp.Pipe(duplex=False) for _ in range(self.num_worker)]
        self.eval_model_pipes = mp.Pipe(duplex=False)
        self.eval_result_pipes = mp.Pipe(duplex=False)

        from model.n4Model.config import Config
        self.config = Config

        args.use_gpu = False
        self.share_agent = agent_class(args, self.config)
        self.share_agent.load_model(args.agent_id)
        self.share_agent.model.share_memory()
        args.use_gpu = True

    def run(self):
        worker_processes = [mp.Process(target=worker_process,
                                       args=(self.share_agent,
                                             self.agent_class,
                                             self.agent_args,
                                             self.env_class,
                                             self.env_config,
                                             self.worker_pipes[worker_id][0],
                                             self.queue))
                            for worker_id in range(self.num_worker)
                            ]

        trainer_process = mp.Process(target=train_process,
                                     args=(self.share_agent,
                                           self.agent_class,
                                           self.agent_args,
                                           [pipe[1] for pipe in self.worker_pipes],
                                           self.queue,
                                           self.eval_model_pipes[1],
                                           self.eval_result_pipes[0]
                                           )
                                     )

        evaler_process = mp.Process(target=eval_process,
                                    args=(self.share_agent,
                                          self.agent_class,
                                          self.agent_args,
                                          self.env_class,
                                          self.env_config,
                                          self.eval_model_pipes[0],
                                          self.eval_result_pipes[1],
                                          64
                                          )
                                    )

        trainer_process.start()
        # time.sleep(10)
        for p in worker_processes:
            p.start()
        # time.sleep(10)
        evaler_process.start()

        import signal

        # 自定义的信号处理函数
        def signal_handler(sig, frame):
            trainer_process.terminate()
            trainer_process.join()
            evaler_process.terminate()
            evaler_process.join()
            for task in worker_processes:
                task.terminate()
                task.join()
            print("Received signal to terminate the program.")
            # 执行必要的清理操作

        # 注册信号处理器
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)  # 也可以处理Ctrl+C终止

        trainer_process.join()
        for p in worker_processes:
            p.join()

        evaler_process.join()
        for pipe in self.worker_pipes:
            pipe[0].close()
            pipe[1].close()
        self.eval_model_pipes[0].close()
        self.eval_model_pipes[1].close()
        self.eval_result_pipes[0].close()
        self.eval_result_pipes[1].close()

        sys.exit(0)  # 正常退出程序


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_worker", type=int, default=1)
    parser.add_argument("--agent_num", type=int, default=1)
    parser.add_argument("--max_agent_num", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--grad_max_norm", type=float, default=0.5)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--returns_norm", type=bool, default=True)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=float, default=512)
    parser.add_argument("--city_nums", type=int, default=20)
    parser.add_argument("--max_city_nums", type=int, default=100)
    parser.add_argument("--allow_back", type=bool, default=False)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=0)
    args = parser.parse_args()

    mp.set_start_method("spawn")

    PW = SharelWorker(Agent, args, MTSPEnv)
    PW.run()
