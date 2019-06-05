import os
from typing import Any, Dict

import numpy as np


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
    def random_loguniform(low, high):
        return lambda: str(np.exp(np.random.uniform(np.log(low), np.log(high))))

    @staticmethod
    def random_subset(*args):
        choices = []
        for arg in args:
            choices.append(arg)
        func = lambda: np.random.choice(choices, np.random.randint(1, len(choices)+1), replace=False)
        return func

    @staticmethod
    def random_pair(*args):
        choices = []
        for arg in args:
            choices.append(arg)
        func = lambda: np.random.choice(choices, 2, replace=False)
        return func

    @staticmethod
    def random_uniform(low, high):
        return lambda: np.random.uniform(low, high)


class HyperparameterSearch:

    def __init__(self, **kwargs):
        self.search_space = {}
        self.lambda_ = lambda: 0
        for key, val in kwargs.items():
            self.search_space[key] = val

    def parse(self, val: Any):
        if isinstance(val, type(self.lambda_)) and val.__name__ == self.lambda_.__name__:
            val = val()
            if isinstance(val, (int, np.int)):
                return int(val)
            elif isinstance(val, (float, np.float)):
                return float(val)
            elif isinstance(val, (np.ndarray, list)):
                return " ".join(val)
            else:
                return val
        elif isinstance(val, (int, np.int)):
            return int(val)
        elif isinstance(val, (float, np.float)):
            return float(val)
        elif isinstance(val, (np.ndarray, list)):
            return " ".join(val)
        elif val is None:
            return None
        else:
            return val


    def sample(self) -> Dict:
        res = {}
        for key, val in self.search_space.items():
            res[key] = self.parse(val)
        return res

    def update_environment(self, sample) -> None:
        for key, val in sample.items():
            os.environ[key] = str(val)
