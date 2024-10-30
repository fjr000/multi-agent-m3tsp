from typing import Tuple, List
import sys
import os

class Config:
    def __init__(self, city_nums: Tuple = (10, 100), agent_nums: Tuple = (1, 9), env_nums: int = 5):
        self.city_nums = city_nums
        self.agent_nums = agent_nums
        self.env_nums = env_nums




