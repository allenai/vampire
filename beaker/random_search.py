from typing import Dict, Any

import numpy as np
import os

class RandomSearch:

    @staticmethod
    def random_choice(*args):
        choices = []
        for arg in args:
            choices.append(arg)
        return lambda: np.random.choice(choices)

    @staticmethod
    def random_integer(low, high):
        return lambda: int(np.random.randint(low, high))

    @staticmethod
    def random_subset(*args):
        choices = []
        for arg in args:
            choices.append(arg)
        subset_length = np.random.randint(1, len(choices)+1)
        return lambda: np.random.choice(choices, subset_length, replace=False)

    @staticmethod
    def random_uniform(low, high):
        return lambda: np.random.uniform(low, high)


class HyperparameterSearch:

    def __init__(self, **kwargs):
        self.search_space = {}
        self.LAMBDA = lambda: 0
        for k, v in kwargs.items():
            self.search_space[k] = v

    def parse(self, val: Any):
        if isinstance(val, (int, np.int)):
            return int(val)
        elif isinstance(val, (float, np.float)):
            return float(val)
        elif isinstance(val, (np.ndarray, list)):
            return ",".join(val)
        elif val is None:
            return None
        elif isinstance(val, type(self.LAMBDA)) and val.__name__ == self.LAMBDA.__name__:
            return val()
        else:
            return val


    def sample(self) -> Dict:
        res = {}
        for k, v in self.search_space.items():
            res[k] = self.parse(v)
        return res

    def update_environment(self, sample) -> None:
        for k, v in sample.items():
            os.environ[k] = str(v)
