import argparse
import copy
import math
import time
import random

import numpy as np
import torch
from typing import Tuple, List, Dict, override
import sys
from envs.MTSP.MTSP5T import MTSPEnv as Env

sys.path.append("../")
from envs.MTSP.Config import Config
from envs.GraphGenerator import GraphGenerator as GG
from utils.GraphPlot import GraphPlot as GP
from model.NNN.RandomAgent import RandomAgent
import torch.nn.functional as F


class MTSPEnv(Env):
    """
    1. 智能体离开仓库后，再次抵达仓库后不允许其离开仓库，salesman mask中只有depot
    2. 允许智能体中当前旅行成本最高M-1个返回仓库，由最后一个智能体保证完备性（还是通过允许最远的返回仓库？）
    3. 允许智能体静止在原地（限制与否？）
    4. 支持batch的环境， batch中单个示例结束后可送入env得到原有的state

    config: dict
        - salesmen  int
        - cities    int
            - salesmen <= cities;
        - mode ['fixed'|'rand'] str
        - seed      int
    """

    def __init__(self, config: Dict = None):
        super(MTSPEnv, self).__init__(config)

    @override
    def step(self, ori_actions: np.ndarray):
        s,r,d,info = super(MTSPEnv, self).step(ori_actions)

        if self.done:
            trajs = info['trajectories']
            last_non_one, last_indices = self.get_last_non_one_vectorized(trajs)
            cur_pos = trajs[...,1:]
            last_pos = trajs[...,:-1]
            penalty = np.zeros_like(trajs)
            # penalty = ((cur_pos != 1) & (cur_pos == last_pos) & (cur_pos != last_non_one[...,np.newaxis])).astype(np.int_)
            # sum_penalty = penalty.sum(axis=-1)
            for i in range(1,trajs.shape[-1]):
                penalty[...,i] = (trajs[...,i] == trajs[...,i-1]) & (trajs[...,i] !=last_non_one) & (i < last_indices)
            penalty = penalty * self._min_distance[:,None,None] * trajs.shape[1] / self.problem_size
            info.update({'penalty':penalty})
            # sum_penalty = penalty.sum(axis=-1)
            # check = np.all(sum_penalty == info['conflict_count'])
            # check_pos = np.argwhere(sum_penalty != info['conflict_count'])

        return s, r, d, info

    def get_last_non_one_vectorized(self, x):
        # x.shape = [B, A, T]
        B,A,T = x.shape
        mask = (x != 1)  # shape: [B, A, T]

        # 反转最后维度，找到第一个非1的位置（即原序列最后一个非1）
        flipped_mask = np.flip(mask, axis=-1)

        # 找到反转后第一个 True 的位置，即原序列中最后一个非1的位置
        last_indices = T - 1 - np.argmax(flipped_mask, axis=-1)

        # 处理全为1的情况（argmax会在0位置返回False），这里将没有非1的设为T
        all_ones = ~np.any(mask, axis=-1)
        last_indices[all_ones] = 0  # 或者设为 None / NaN 等

        # 使用高级索引提取对应值
        B, A, T = x.shape
        result = x[np.arange(B)[:, None, None],
        np.arange(A)[None, :, None],
        last_indices[..., None]]

        return result.reshape(B, A), last_indices


def test2Env(agent, env1, env2):

    agent_num = 5
    city_num = 50
    batch_size = 32

    g = GG()
    batch_graph = g.generate(batch_size=batch_size, num=city_num)
    batch_graph = GG.augment_xy_data_by_8_fold_numpy(batch_graph)

    env_config = {
        "salesmen": agent_num,
        "cities": city_num,
        "seed": None,
        "mode": 'rand',
        "env_masks_mode": 7,
        "use_conflict_model": True
    }


    states1, env_info1 = env1.reset(env_config, batch_graph)
    states2, env_info2 = env2.reset(env_config, batch_graph)

    assert np.isclose(env1.graph_matrix.cpu().numpy(), env2.graph_matrix).all()
    if not np.isclose(states1, states2).all():
        x = np.argwhere(~np.isclose(states1,states2))
        aa  = x.nonzero()
        pass
    assert np.isclose(states1, states2).all()



    def assert_check(d,d2):
        for k,v in d.items():
            if isinstance(v,np.ndarray):
                assert np.isclose(v, d2[k]).all(), f"k:{k}"
            else:
                print(f"k:{k}:->{v}, -> {d2[k]}" )
                assert v == d2[k], f"k:{k}"
    assert_check(env_info2,env_info1)

    states = states1
    env_info = env_info1

    salesmen_masks = env_info["salesmen_masks"]
    masks_in_salesmen = env_info["masks_in_salesmen"]
    city_mask = env_info["mask"]
    dones = env_info["dones"]

    agent.reset_graph(batch_graph, agent_num)
    info = {
        "use_conflict_model":True
    }


    done = False
    while not done:
        states_t = _convert_tensor(states, device=agent.device)
        # mask: true :not allow  false:allow

        salesmen_masks_t = _convert_tensor(~salesmen_masks, dtype=torch.bool, device=agent.device)
        if masks_in_salesmen is not None:
            masks_in_salesmen_t = _convert_tensor(~masks_in_salesmen, dtype=torch.bool, device=agent.device)
        else:
            masks_in_salesmen_t = None
        city_mask_t = _convert_tensor(~city_mask, dtype=torch.bool, device=agent.device)
        dones_t = _convert_tensor(dones, dtype=torch.bool, device=agent.device) if dones is not None else None
        info.update({
            "masks_in_salesmen": masks_in_salesmen_t,
            "mask": city_mask_t,
            "dones": dones_t
        })

        acts, act_logp, act_nf, act_entropy = agent.predict(states_t, salesmen_masks_t, info)
        if act_nf is not None:
            states, r, done, env_info = env1.step(act_nf + 1)
        else:
            states, r, done, env_info = env1.step(acts + 1)

        if act_nf is not None:
            states2, r2, done2, env_info2 = env2.step(act_nf + 1)
        else:
            states2, r2, done2, env_info2 = env2.step(acts + 1)

        x = np.isclose(states2, states)
        xx = np.argwhere(~x)

        if not np.isclose(states2, states).all():
            xx = np.argwhere(~np.isclose(states2, states))
            pass
        assert np.isclose(r, r2).all()
        assert done2 == done
        assert_check(env_info,env_info2)

        salesmen_masks = env_info["salesmen_masks"]
        masks_in_salesmen = env_info["masks_in_salesmen"]
        city_mask = env_info["mask"]
        dones = env_info["dones"]




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--agent_num", type=int, default=4)
    parser.add_argument("--fixed_agent_num", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad_max_norm", type=float, default=0.5)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=False)
    parser.add_argument("--max_ent", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=0)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--augment", type=int, default=8)
    parser.add_argument("--repeat_times", type=int, default=1)
    parser.add_argument("--city_nums", type=int, default=5)
    parser.add_argument("--random_city_num", type=bool, default=False)
    parser.add_argument("--model_dir", type=str, default="../pth/")
    parser.add_argument("--agent_id", type=int, default=999999999)
    parser.add_argument("--env_masks_mode", type=int, default=7,
                        help="0 for only the min cost  not allow back depot; 1 for only the max cost allow back depot")
    parser.add_argument("--eval_interval", type=int, default=400, help="eval  interval")
    parser.add_argument("--use_conflict_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_conflict_model", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--train_actions_model", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--train_city_encoder", type=bool, default=True, help="0:not use;1:use")
    parser.add_argument("--use_agents_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--use_city_mask", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--agents_adv_rate", type=float, default=0.0, help="rate of adv between agents")
    parser.add_argument("--conflict_loss_rate", type=float, default=1.0, help="rate of adv between agents")
    parser.add_argument("--only_one_instance", type=bool, default=False, help="0:not use;1:use")
    parser.add_argument("--save_model_interval", type=int, default=800, help="save model interval")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--epoch_size", type=int, default=1280000, help="number of instance for each epoch")
    parser.add_argument("--n_epoch", type=int, default=100, help="number of epoch")
    args = parser.parse_args()
    def set_seed(seed=42):
        # 基础库
        random.seed(seed)
        np.random.seed(seed)

        # PyTorch核心设置
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU时
        # # 禁用CUDA不确定算法
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    g = GG()
    batch_graph = g.generate(batch_size=args.batch_size, num=args.city_nums)
    batch_graph = GG.augment_xy_data_by_8_fold_numpy(batch_graph)

    set_seed()
    env_config = {
        "salesmen": args.agent_num,
        "cities": args.city_nums,
        "seed": None,
        "mode": 'rand',
        "env_masks_mode": args.env_masks_mode,
        "use_conflict_model": args.use_conflict_model
    }

    env = MTSPEnv(
        env_config
    )

    from envs.GraphGenerator import GraphGenerator as GG

    # g = GG(1, env_config["cities"])
    # graph = g.generate(1, env_config["cities"], dim=2)

    from algorithm.Attn.AgentV1 import Agent as Agent
    from model.AttnModel.Model import Config

    agent = Agent(args, Config)
    agent.load_model(args.agent_id)
    # features_nb, actions_nb, actions_no_conflict_nb, returns_nb, individual_returns_nb, masks_nb, dones_nb = agent.run_batch(
    #     env, graph, env_config["salesmen"], 32)
    from utils.TensorTools import _convert_tensor
    import numpy as np
    import torch
    from envs.GraphGenerator import GraphGenerator as GG
    import time

    start_time = time.time()
    # for i in range(100):
    #     o1 = agent.run_batch_episode(env, batch_graph, args.agent_num, True, info={
    #         "use_conflict_model": args.use_conflict_model})
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))
    # set_seed()
    from envs.MTSP.MTSP5_Penalty import MTSPEnv as Env4
    env4 = Env4(env_config)
    #
    # agent = Agent(args, Config)
    # agent.load_model(args.agent_id)
    #
    # start_time = time.time()
    # for i in range(1):
    #     o2 = agent.run_batch_episode(env4, batch_graph, args.agent_num, True, info={
    #         "use_conflict_model": args.use_conflict_model})
    # end_time = time.time()
    # print("--- %s seconds ---" % (end_time - start_time))
    test2Env(agent,env,env4)