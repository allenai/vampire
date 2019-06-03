[![codecov](https://codecov.io/gh/allenai/vae/branch/master/graph/badge.svg?token=NOriF2Rm8p)](https://codecov.io/gh/allenai/vae)

# vampire

*Exploring Variational Autoencoders for Representation Learning in NLP*


## Installation

Install necessary dependencies via `requirements.txt`, which will include the latest unreleased install of `allennlp` (from the `master` branch).

```
$ pip install -r requirements.txt
```

Verify your installation by running: 

```
$ SEED=42 pytest -v --color=yes vampire
```

All tests should pass.

## Download Data

Download your dataset of interest, and make sure it is made up of json files, where each line of each file corresponds to a separate instance. Each line must contain a `text` field, and optionally a `label` field. 

In this tutorial we use the AG News dataset hosted on AllenNLP.

```
$ mkdir -p examples/ag
$ curl -Lo examples/ag/train.jsonl https://s3-us-west-2.amazonaws.com/allennlp/datasets/ag-news/train.jsonl
$ curl -Lo examples/ag/dev.jsonl https://s3-us-west-2.amazonaws.com/allennlp/datasets/ag-news/dev.jsonl
```

## Preprocess data

To make pretraining fast, we precompute fixed bag-of-words representations of the data. 

```
$ python -m scripts.preprocess_data --train-path examples/ag/train.jsonl --dev-path examples/ag/dev.jsonl --tokenize --tokenizer-type spacy --vocab-size 10000 --serialization-dir examples/ag
```

This script will tokenize your data, and save the resulting output into the specified `serialization-dir`.

In `examples/ag`, you should see:
    1. `train.npz` - pre-computed bag of word representations of the training data
    2. `dev.npz` - pre-computed bag of word representations of the dev data
    3. `vampire.bgfreq` - background frequencies of words used to bias VAMPIRE reconstruction during training
    4. `vocabulary` - AllenNLP vocabulary directory

## Create reference corpus

We have to build a reference corpus to calcuate NPMI (normalized pointwise mutual information), a measure of topical coherence that we use for early stopping.

We use the validation data as our reference corpus. Run:

```
$ python -m scripts.make_reference_corpus examples/ag/dev.jsonl examples/ag/reference
```

In `examples/ag/reference`, you should see:
    1. `ref.npz` - pre-computed bag of word representations of the reference corpus
    2. `ref.vocab.json` - the reference corpus vocabulary


## Pre-train VAMPIRE

Set your data directory as the environment variable:

```
$ export DATA_DIR="$(pwd)/examples/ag"
```

Then run VAMPIRE:

```
$ python -m scripts.train -c ./training_config/vampire.jsonnet -s ./model_logs/vampire -e VAMPIRE --device -1
```

To run on a GPU, run with `--device 0` (or any other available GPU device number)

This command will output model_logs at `./model_logs/vampire` from the training config `./training_config/vampire.jsonnet`. 

For convenience, you can run the train script with the `--override` flag to remove the previous experiment at the same serialization directory.


## Use VAMPIRE with downstream classifier

Go into `./training_config/classifier.jsonnet`, and rename filepaths accordingly.

You can change the `VAE_FIELDS` in the `classifier.jsonnet` to your newly trained VAE:

```
local VAE_FIELDS = {
    "vae_indexer": {
        "vae_tokens": {
            "type": "single_id",
            "namespace": "vae",
            "lowercase_tokens": true
        }
    },  
    "vae_embedder": {
        "vae_tokens": {
                "type": "vae_token_embedder",
                "expand_dim": true,
                "model_archive": "/path/to/model_logs/vae_unsupervised/model.tar.gz",
                "background_frequency": "/path/to/model_logs/vae_unsupervised/vocabulary/vae.bgfreq.json",
                "dropout": 0.2
        }
    }
};
```

*Note* : You can additionally subsample the training data by setting `{"dataset_reader": {"sample": N}}` where `N < len(train.jsonl)`.


Run

```
$ python -m scripts.train -x 42 -c ./training_config/jsonnet/classifier.jsonnet -s ./model_logs/boe --override -e BOE
```

This command will output model_logs at `./model_logs/boe` from the training config `./training_config/local/classifier.jsonnet`. The `override` flag will override previous experiment at the same serialization directory.

