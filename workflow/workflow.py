from model.TestModel import TestModel as Model
from envs.MTSP.MTSP import MTSPEnv as Env
import numpy as np
from utils.ParallelSampler import ParallelSamplerAsync as ParallelSampler

def get_cur_idxs(agents_trajectory):
    return np.array(agents_trajectory[:, -1], dtype=np.int64)


def get_last_idxs(agents_trajectory):
    last_idxs = []
    for traj in agents_trajectory:
        if len(traj) > 1:
            last_idxs.append(traj[-2])
        else:
            last_idxs.append(traj[-1])
    return np.array(last_idxs, dtype=np.int64)


def city_parse(graph, global_mask, last_idxs, cur_idxs):
    depot = graph[0]
    last_visited_city = graph[last_idxs - 1]
    cur_visited_city = graph[cur_idxs - 1]

    to_visited_cities_idxs = np.argwhere(global_mask == 0).squeeze(-1)
    to_visited_city = graph[to_visited_cities_idxs]

    return [depot, last_visited_city, cur_visited_city, to_visited_city], [np.ones((1,)), last_idxs, cur_idxs,
                                                                           to_visited_cities_idxs + 1]


if __name__ == '__main__':
    num_worker = 16
    sample_times = 100
    model = Model(3,2,128)
    Config = {
        "city_nums": (20, 40),
        "agent_nums": (1, 4),
        "seed": 1111,
        "fixed_graph": False,
        "allow_back": True
    }
    PS = ParallelSampler(model, Env,num_worker,Config)

    for t in range(sample_times):
        PS.start()
        obs_lists, reward_lists, done_lists, global_mask_lists, action_mask_lists, global_info_list = PS.collect()
        # model update
        PS.update_agent(t + 1, model)
    PS.close()
    # # for i in range(num_worker):
    # #     global_info = global_info_list[i]
    # #     gp.draw_route(global_info["graph"], global_info["actors_trajectory"], one_first=True)
    # print(f"time_cost_per_worker:{(ed-st)/1e9 / num_worker / sample_times}")

