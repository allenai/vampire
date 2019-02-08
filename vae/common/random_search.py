from typing import Dict

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
        for k, v in kwargs.items():
            self.search_space[k] = v

    def sample(self) -> Dict:
        res = {}
        for k, v in self.search_space.items():
            if isinstance(v, int) or isinstance(v, np.int):
                res[k] = v
            elif isinstance(v, str):
                res[k] = v
            elif isinstance(v, np.ndarray) or isinstance(v, list):
                res[k] = ",".join(v)
            else:
                val = v()
                if isinstance(val, np.int):
                    val = int(val)
                elif isinstance(val, np.float):
                    val = float(val)
                elif isinstance(val, np.ndarray) or isinstance(val, list):
                    res[k] = ",".join(val)
                else:
                    res[k] = val
        return res

    def update_environment(self, sample) -> None:
        for k, v in sample.items():
            os.environ[k] = str(v)
