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

def run_with_beaker(param_file, config, args):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], universal_newlines=True).strip()
    image = f"vampire"

    print(f"Building the Docker image ({image})...")
    subprocess.run(f'docker build -t {image} .', shell=True, check=True)

    print(f"Creating a Beaker image...")
    _ = subprocess.check_output(f'beaker image create {image}', shell=True, universal_newlines=True).strip()
    print(f"  image created: {image}")

    # If the git repository is dirty, add a random hash.
    # result = subprocess.run('git diff-index --quiet HEAD --', shell=True)
    # if result.returncode != 0:
        
    #     dirty_hash = "%x" % random_int
    #     image += "-" + dirty_hash

    config_tasks = []
    # Reads params and sets environment.
    ext_vars = {}
    overrides = ""
    ext_vars = {}
    for k, v in config.items():
        ext_vars[k] = str(v)

    # for var in args.env:
    #     key, value = var.split("=")
    #     ext_vars[key] = value
    params = Params.from_file(param_file, overrides, ext_vars)
    flat_params = params.as_flat_dict()
    env: Dict[str, Any] = {}
    for k, v in flat_params.items():
        k = str(k).replace('.', '_')
        if isinstance(v, bool):
            env[k] = v
        else:
            env[k] = str(v)

    config_dataset_id = subprocess.check_output(f'beaker dataset create --quiet {param_file}',
                                                shell=True,
                                                universal_newlines=True).strip()
    allennlp_command = [
            "-m",
            "allennlp.run",
            "train",
            "/config.json",
            "-s",
            "/output",
            "--file-friendly-logging",
            "--include-package",
            "vampire"
        ]

    dataset_mounts = []
    for source in [f"{config_dataset_id}:/config.json"]:
        datasetId, containerPath = source.split(":")
            "datasetId": "datasets/vampire",
            "containerPath": containerPath
        })

    # for var in args.env:
    #     key, value = var.split("=")
    #     env[key] = value

    for k, v in config.items():
        env[k] = str(v)

    # requirements = {}
    # if args.cpu:
    #     requirements["cpu"] = float(args.cpu)
    # if args.memory:
    #     requirements["memory"] = args.memory
    # if args.gpu_count:
    #     requirements["gpuCount"] = int(args.gpu_count)
    config_spec = {
        "description": "",
        "image": image,
        "resultPath": "/output",
        "args": allennlp_command,
        "datasetMounts": dataset_mounts,
        "requirements": {},
        "env": env
    }
    config_task = {"spec": config_spec, "name": f"training_0"}

    config = {
        "tasks": [config_task]
    }

    output_path = tempfile.mkstemp(".yaml", "beaker-config-")[1]
    with open(output_path, "w") as output:
        output.write(json.dumps(config, indent=4))
    print(f"Beaker spec written to {output_path}.")

    experiment_command = ["beaker", "experiment", "create", "--file", output_path]
    # if args.name:
    #     experiment_command.append("--name")
    #     experiment_command.append(args.name.replace(" ", "-"))

    # if args.dry_run:
    #     print(f"This is a dry run (--dry-run).  Launch your job with the following command:")
    #     print(f"    " + " ".join(experiment_command))
    # else:
    print(f"Running the experiment:")
    print(f"    " + " ".join(experiment_command))
    subprocess.run(experiment_command)

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
    parser.add_argument('-b', '--beaker', action='store_true', help = "run on beaker")

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

    if args.beaker:
        run_with_beaker(args.config, sample, args)
    else:
        subprocess.run(" ".join(allennlp_command), shell=True, check=True)


if __name__ == '__main__':
    main()
