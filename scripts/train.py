import os
import argparse
import subprocess
import shutil
from beaker.search_environments import SEARCH_ENVIRONMENTS
from beaker.random_search import HyperparameterSearch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument('-o',
                        '--override',
                        action="store_true",
                        help='path to directory containing reference counts and vocab')
    parser.add_argument('-x',
                        '--seed',
                        type=int,
                        help='seed',
                        required=False,
                        default=42)
    parser.add_argument('-c', '--config', type=str, help='training config', required=True)
    parser.add_argument('-s', '--serialization_dir', type=str, help='model serialization directory', required=True)

    parser.add_argument('-e', '--environment', type=str, help='environment', required=True)

    args = parser.parse_args()

    os.environ['SEED'] = str(args.seed)

    if os.path.exists(args.serialization_dir) and args.override:
        print(f"overriding {args.serialization_dir}")
        shutil.rmtree(args.serialization_dir)

    environment = SEARCH_ENVIRONMENTS[args.environment]
    
    search_space = HyperparameterSearch(**environment)

    sample = search_space.sample()
    for key, val in sample.items():
        os.environ[key] = str(val)

    allennlp_command = [
                "allennlp",
                "train",
                "--include-package",
                "vae",
                args.config,
                "-s",
                args.serialization_dir,
            ]
    subprocess.run(" ".join(allennlp_command), shell=True, check=True)
