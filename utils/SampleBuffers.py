import numpy as np
import random

class Config:
    buffer_size = 100000
    batch_size = 128
    agents_num = (2,5)
    cities_num = (20,50)
    agent_dim = 3
    city_dim = 2

class SampleBuffers:
    def __init__(self, config):
        self.buffer_size = config.buffer_size
        self.batch_size = config.batch_size
        self.min_agent_num = config.agents_num[0]
        self.max_agent_num = config.agents_num[1]
        self.min_cities_num = config.cities_num[0]
        self.max_cities_num = config.cities_num[1]
        self.agent_dim = config.agent_dim
        self.city_dim = config.city_dim

        self.max_enum_size = (self.max_agent_num - self.min_agent_num +1) * (self.max_cities_num - self.min_cities_num + 1)

        self.buffer_ptr = np.zeros(self.max_enum_size)
        self.buffer=[
            {
                "agents_state": np.zeros((self.buffer_size, self.__get_agents_num(idx), self.agent_dim), dtype=np.float32),
                "cities_state": np.zeros((self.buffer_size, self.__get_cities_num(idx), self.city_dim), dtype=np.float32),
                "global_mask": np.zeros((self.buffer_size, self.__get_cities_num(idx), 1), dtype=np.float32),
                "agents_action_mask": np.zeros((self.buffer_size, self.__get_agents_num(idx), 2), dtype=np.float32),
                "next_agents_state":np.zeros((self.buffer_size, self.__get_agents_num(idx), self.agent_dim), dtype=np.float32),
                "next_global_mask":np.zeros((self.buffer_size, self.__get_cities_num(idx), 1), dtype=np.float32),
                "next_agents_action_mask":np.zeros((self.buffer_size, self.__get_agents_num(idx), 2), dtype=np.float32),
                "agents_reward": np.zeros((self.buffer_size, self.__get_agents_num(idx), 1), dtype=np.float32),
                "done": np.zeros((self.buffer_size, 1), dtype=np.float32)
            }

            for idx in range(self.max_enum_size)
        ]

    def __get_index(self, agents_num, cities_num):
        assert self.min_agent_num <= agents_num <= self.max_agent_num, "agents_num must be between {} and {}".format(self.min_agent_num, self.max_agent_num)
        assert self.min_cities_num <= cities_num <= self.max_cities_num , "city_num must be between {} and {}".format(self.min_cities_num, self.max_cities_num)

        return (agents_num - self.min_agent_num) * (self.max_cities_num - self.min_cities_num + 1) + cities_num - self.min_cities_num

    def __get_agents_num(self, index):
        return index // (self.max_cities_num - self.min_cities_num + 1)

    def __get_cities_num(self, index):
        return index % (self.max_cities_num - self.min_cities_num + 1)

    def store_one(self, agents_state, cities_state, global_mask, agents_action_mask, agents_reward, done,
                  next_agents_state, next_global_mask, next_agents_action_mask):
        agent_num = len(agents_state)
        city_num = len(cities_state)
        idx = self.__get_index(agent_num, city_num)
        ptr = self.buffer_ptr[idx]
        self.buffer[idx]["agents_state"][ptr] = agents_state
        self.buffer[idx]["agents_reward"][ptr] = agents_reward
        self.buffer[idx]["cities_state"][ptr] = cities_state
        self.buffer[idx]["done"][ptr] = done
        self.buffer[idx]["global_mask"][ptr] = global_mask
        self.buffer[idx]["agents_action_mask"][ptr] = agents_action_mask

        self.buffer_ptr[idx] = (self.buffer_ptr[idx] + 1) % self.buffer_size

    def sample(self, agent_num = None, city_num= None):
        if agent_num is not None:
            assert self.min_agent_num <= agent_num <= self.max_agent_num, "agent_num must be between {} and {}".format(self.min_agent_num, self.max_agent_num)
        if city_num is not None:
            assert self.min_cities_num <= city_num <= self.max_cities_num, "city_num must be between {} and {}".format(self.min_cities_num, self.max_cities_num)

        idx_to_choose = np.argwhere(self.buffer_ptr >= self.buffer_size)
        if len(idx_to_choose) == 0:
            return None
        else:
            idx = random.choice(idx_to_choose)
            sample_idxs = np.random.randint(0,self.buffer_ptr[idx],size=self.batch_size)
            agent_state = self.buffer[idx]["agents_state"][sample_idxs]
            agents_action_mask = self.buffer[idx]["agents_action_mask"][sample_idxs]
            cities_state = self.buffer[idx]["cities_state"][sample_idxs]
            agent_reward = self.buffer[idx]["agents_reward"][sample_idxs]
            done = self.buffer[idx]["done"][sample_idxs]
            global_mask = self.buffer[idx]["global_mask"][sample_idxs]
            return agent_state, agents_action_mask, agent_reward, cities_state, done, global_mask


if __name__ == '__main__':
    SBS = SampleBuffers(Config())
    from envs.MTSP.MTSP import MTSPEnv
    env = MTSPEnv()
    states, info = env.reset()
    anum = info["anum"]
    cnum = info["cnum"]
    graph = info["graph"]
    global_mask = info["global_mask"]
    agents_action_mask = info["agents_action_mask"]




    pass