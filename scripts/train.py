import argparse
import os
import shutil
import subprocess
from environments import ENVIRONMENTS
from environments.random_search import HyperparameterSearch

def main():
    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument('-o',
                        '--override',
                        action="store_true",
                        help='remove the specified serialization dir before training')
    parser.add_argument('-c', '--config', type=str, help='training config', required=True)
    parser.add_argument('-s', '--serialization_dir', type=str, help='model serialization directory', required=True)
    parser.add_argument('-e', '--environment', type=str, help='hyperparameter environment', required=True)
    parser.add_argument('-r', '--recover', action='store_true', help = "recover saved model")
    parser.add_argument('-x', '--cpu', action='store_true', help = "run on cpu")

    args = parser.parse_args()

    env = ENVIRONMENTS[args.environment.upper()]

    if args.cpu:
        env['CUDA_DEVICE'] = -1

    space = HyperparameterSearch(**env)

    sample = space.sample()

    for key, val in sample.items():
        os.environ[key] = str(val)
    
    if os.path.exists(args.serialization_dir) and args.override:
        print(f"overriding {args.serialization_dir}")
        shutil.rmtree(args.serialization_dir)

    allennlp_command = [
            "allennlp",
            "train",
            "--include-package",
            "vampire",
            args.config,
            "-s",
            args.serialization_dir,
            ]
    if args.recover:
        allennlp_command.append("--recover")
    subprocess.run(" ".join(allennlp_command), shell=True, check=True)


if __name__ == '__main__':
    main()
