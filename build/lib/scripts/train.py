import argparse
import json
import os
import random
import shutil
import subprocess
import tempfile
from typing import Any, Dict

from allennlp.common.params import Params

from environments import ENVIRONMENTS
from environments.random_search import HyperparameterSearch

random_int = random.randint(0, 2**32)

def main():
    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument('-o',
                        '--override',
                        action="store_true",
                        help='remove the specified serialization dir before training')
    parser.add_argument('-c', '--config', type=str, help='training config', required=True)
    parser.add_argument('-s', '--serialization-dir', type=str, help='model serialization directory', required=True)
    parser.add_argument('-e', '--environment', type=str, help='hyperparameter environment', required=True)
    parser.add_argument('-r', '--recover', action='store_true', help = "recover saved model")
    parser.add_argument('-d', '--device', type=str, required=False, help = "device to run model on")
    parser.add_argument('-x', '--seed', type=str, required=False, help = "seed to run on")


    args = parser.parse_args()

    env = ENVIRONMENTS[args.environment.upper()]


    space = HyperparameterSearch(**env)

    sample = space.sample()

    for key, val in sample.items():
        os.environ[key] = str(val)

    if args.device:
        os.environ['CUDA_DEVICE'] = args.device

    if args.seed:
        os.environ['SEED'] = args.seed


    allennlp_command = [
            "allennlp",
            "train",
            "--include-package",
            "vampire",
            args.config,
            "-s",
            args.serialization_dir
            ]

    if args.seed:
        allennlp_command[-1] = allennlp_command[-1] + "_" + args.seed

    if args.recover:
        def append_seed_to_config(seed, serialization_dir):
            seed = str(seed)
            seed_dict = {"pytorch_seed": seed,
                         "random_seed": seed,
                         "numpy_seed": seed}
            config_path = os.path.join(serialization_dir, 'config.json')
            with open(config_path, 'r+') as f:
                config_dict = json.load(f)
                seed_dict.update(config_dict)
                f.seek(0)
                json.dump(seed_dict, f, indent=4)

        append_seed_to_config(seed=args.seed, serialization_dir=allennlp_command[-1])

        allennlp_command.append("--recover")

    if os.path.exists(allennlp_command[-1]) and args.override:
        print(f"overriding {allennlp_command[-1]}")
        shutil.rmtree(allennlp_command[-1])


    subprocess.run(" ".join(allennlp_command), shell=True, check=True)


if __name__ == '__main__':
    main()
