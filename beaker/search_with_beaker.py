#! /usr/bin/env python

# Script to launch AllenNLP Beaker jobs.

import argparse
import os
import json
import random
import tempfile
import subprocess
import sys
from typing import List, Dict, Any
import numpy as np
from vae.common.util import read_json
from tqdm import tqdm
# This has to happen before we import spacy (even indirectly), because for some crazy reason spacy
# thought it was a good idea to set the random seed on import...
random_int = random.randint(0, 2**32)

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(os.path.join(__file__, os.pardir), os.pardir))))

from allennlp.common.params import Params


def step():
    hidden_dim = int(np.random.choice([512, 1024, 2048]))
    latent_dim = int(np.random.choice([50, 128, 256, 512]))
    encoder_layers = int(np.random.choice([2, 3]))
    kl_weight_annealing = np.random.choice(['linear', 'sigmoid'])

    return {
        "model.vae.encoder.hidden_dims": [hidden_dim] * encoder_layers,
        "model.vae.mean_projection.input_dim": hidden_dim,
        "model.vae.mean_projection.hidden_dims": [latent_dim],
        "model.vae.log_variance_projection.input_dim": hidden_dim,
        "model.vae.log_variance_projection.hidden_dims": [latent_dim],
        "model.vae.decoder.input_dim": latent_dim,
        "model.vae.encoder.activations": ['softplus'] * encoder_layers,
        "model.kl_weight_annealing": kl_weight_annealing,
        "model.vae.encoder.num_layers": encoder_layers,
    }

def classifier_step():

    glove_embeddings = [
      (50,  "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.50d.txt.gz"),
      (100, "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz"),
      (300, "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz")
    ]
    embedding_choice = np.random.choice(len(glove_embeddings), 1)[0]
    embedding_dim, pretrained_file = glove_embeddings[embedding_choice]

    sample = {
        "model.input_embedder.token_embedders.tokens.embedding_dim": embedding_dim,
        "model.input_embedder.token_embedders.tokens.pretrained_file": pretrained_file,
    }

    hidden_dim = embedding_dim
    encoder_type = np.random.choice(["boe", "cnn", "lstm"])

    if encoder_type == "boe":
        encoder_sample = {
            "model.encoder.type": "boe",
            "model.encoder.embedding_dim": embedding_dim,
        }
    else:
        hidden_dim = np.random.uniform(128, 512)
        if encoder_type == "cnn":
            encoder_sample = {
                "model.encoder.type": "cnn",
                "model.encoder.num_filters": np.random.uniform(8, 64),
                "model.encoder.embedding_dim": embedding_dim,
                "model.encoder.output_dim": hidden_dim
            }
        else:
            encoder_sample = {
                "model.encoder.type": "lstm",
                "model.encoder.input_size": embedding_dim,
                "model.encoder.num_layers": np.random.uniform(1, 5)
            }

    classifier = {
        "model.classification_layer.input_dim": hidden_dim,
    }

    sample = sample.update(encoder_sample)
    sample = sample.update(classifier)

    return sample


def generate_json(num_samples: int, include_classifier: bool=False):
    res = []
    for _ in range(num_samples):
        sample = step()
        if include_classifier:
            sample = sample.update(classifier_step())
        res.append(json.dumps(sample))
    return res

def main(param_file: str, overrides: List[str], args: argparse.Namespace):

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], universal_newlines=True).strip()
    image = f"allennlp/allennlp:{commit}"

    if args.blueprint:
            blueprint = args.blueprint
            print(f"Using the specified blueprint: {blueprint}")
    else:
        print(f"Building the Docker image ({image})...")
        subprocess.run(f'docker build -t {image} .', shell=True, check=True)

        print(f"Create a Beaker blueprint...")
        blueprint = subprocess.check_output(f'beaker blueprint create --quiet {image}', shell=True, universal_newlines=True).strip()
        print(f"  Blueprint created: {blueprint}")

    # If the git repository is dirty, add a random hash.
    result = subprocess.run('git diff-index --quiet HEAD --', shell=True)
    if result.returncode != 0:
        dirty_hash = "%x" % random_int
        image += "-" + dirty_hash

    config_tasks = []
    param_files = []
    for ix, override in enumerate(overrides):
        params = Params.from_file(param_file, override, {})

        output_file = os.path.join("/tmp", f"{ix}.config")
        params.to_file(output_file)
        param_files.append(output_file)

    for ix, param_file in tqdm(enumerate(param_files), total=len(param_files)):
        # Reads params and sets environment.
        ext_vars = {}

        for var in args.env:
            key, value = var.split("=")
            ext_vars[key] = value
        params = Params.from_file(param_file, "", ext_vars)
        flat_params = params.as_flat_dict()
        env: Dict[str, Any] = {}
        for k, v in flat_params.items():
            k = str(k).replace('.', '_')
            if isinstance(v, bool):
                env[k] = v
            else:
                env[k] = str(v)

        config_dataset_id = subprocess.check_output(f'beaker dataset create --quiet {param_file}', shell=True, universal_newlines=True).strip()
        allennlp_command = [
                "-m",
                "allennlp.run",
                "train",
                "/config.json",
                "-s",
                "/output",
                "--file-friendly-logging",
                "--include-package",
                "vae.models.unsupervised",
                "--include-package",
                "vae.models.joint_semi_supervised",
                "--include-package",
                "vae.data.dataset_readers.semisupervised_text_classification_json",
                "--include-package",
                "vae.data.tokenizers.regex_and_stopword_filter",
                "--include-package",
                "vae.common.allennlp_bridge"
            ]

        dataset_mounts = []
        for source in args.source + [f"{config_dataset_id}:/config.json"]:
            datasetId, containerPath = source.split(":")
            dataset_mounts.append({
                "datasetId": datasetId,
                "containerPath": containerPath
            })

        for var in args.env:
            key, value = var.split("=")
            env[key] = value

        requirements = {}
        if args.cpu:
            requirements["cpu"] = float(args.cpu)
        if args.memory:
            requirements["memory"] = args.memory
        if args.gpu_count:
            requirements["gpuCount"] = int(args.gpu_count)
        config_spec = {
            "description": args.desc,
            "blueprint": blueprint,
            "resultPath": "/output",
            "args": allennlp_command,
            "datasetMounts": dataset_mounts,
            "requirements": requirements,
            "env": env
        }
        config_task = {"spec": config_spec, "name": f"training_{ix}"}
        config_tasks.append(config_task)

    config = {
        "tasks": config_tasks
    }

    output_path = args.spec_output_path if args.spec_output_path else tempfile.mkstemp(".yaml",
            "beaker-config-")[1]
    with open(output_path, "w") as output:
        output.write(json.dumps(config, indent=4))
    print(f"Beaker spec written to {output_path}.")

    experiment_command = ["beaker", "experiment", "create", "--file", output_path]
    if args.name:
        experiment_command.append("--name")
        experiment_command.append(args.name.replace(" ", "-"))

    if args.dry_run:
        print(f"This is a dry run (--dry-run).  Launch your job with the following command:")
        print(f"    " + " ".join(experiment_command))
    else:
        print(f"Running the experiment:")
        print(f"    " + " ".join(experiment_command))
        subprocess.run(experiment_command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('param_file', type=str, help='The model configuration file.')
    parser.add_argument('--num-samples', type=int, help='number of times to sample during search')
    parser.add_argument('--name', type=str, help='A name for the experiment.')
    parser.add_argument('--spec_output_path', type=str, help='The destination to write the experiment spec.')
    parser.add_argument('--dry-run', action='store_true', help='If specified, an experiment will not be created.')
    parser.add_argument('--blueprint', type=str, help='The Blueprint to use (if unspecified one will be built)')
    parser.add_argument('--desc', type=str, help='A description for the experiment.')
    parser.add_argument('--env', action='append', default=[], help='Set environment variables (e.g. NAME=value or NAME)')
    parser.add_argument('--source', action='append', default=[], help='Bind a remote data source (e.g. source-id:/target/path)')
    parser.add_argument('--cpu', help='CPUs to reserve for this experiment (e.g., 0.5)')
    parser.add_argument('--gpu-count', default=1, help='GPUs to use for this experiment (e.g., 1 (default))')
    parser.add_argument('--memory', help='Memory to reserve for this experiment (e.g., 1GB)')
    parser.add_argument('--include-classifier', default=False, action='store_true', help='If provided, will include random parameters for a joint classifier.')

    args = parser.parse_args()

    overrides = generate_json(args.num_samples, args.include_classifier)

    main(args.param_file, overrides, args)
