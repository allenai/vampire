# VAMPIRE <img src="figures/bat.png" width="60"> 

VAriational Methods for Pretraining In Resource-limited Environments

Read paper [here](https://arxiv.org/abs/1906.02242).

## Citation

```
@inproceedings{vampire,
 author = {Suchin Gururangan and Tam Dang and Dallas Card and Noah A. Smith},
 title = {Variational Pretraining for Semi-supervised Text Classification},
 year = {2019},
 booktitle = {Proceedings of ACL},
}
```


## Installation

Install necessary dependencies via `requirements.txt`, which will include the latest unreleased install of `allennlp` (from the `master` branch).

```
pip install -r requirements.txt
```

Install the spacy english model with:

```
python -m spacy download en
```

Verify your installation by running: 

```
SEED=42 pytest -v --color=yes vampire
```

All tests should pass.


## Install from Docker

Alternatively, you can install the repository with Docker.

First, build the container: 

```
docker build -f Dockerfile --tag vampire/vampire:latest .
```

Then, run the container:

```
docker run -it vampire/vampire:latest
```

This will open a shell in a docker container that has all the dependencies installed.

## Download Data

Download your dataset of interest, and make sure it is made up of json files, where each line of each file corresponds to a separate instance. Each line must contain a `text` field, and optionally a `label` field. 

In this tutorial we use the AG News dataset hosted on AllenNLP. Download it using the following script:

```
sh scripts/download_ag.sh
```

This will make an `examples/ag` directory with train, dev, test files from the AG News corpus.t

## Preprocess data

To make pretraining fast, we precompute fixed bag-of-words representations of the data. 

```
python -m scripts.preprocess_data \
            --train-path examples/ag/train.jsonl \
            --dev-path examples/ag/dev.jsonl \
            --tokenize \
            --vocab-size 30000 \
            --serialization-dir examples/ag
```

You can also precompute BPE, byte-level BPE or BERT tokenizations of the data by first training a tokenizer:

``` 
jq -r '.text' examples/ag/train.jsonl > train.txt  # use the jq library to get the raw training text
python -m scripts.train_tokenizer --input_file train.txt \
            --tokenizer_type BERT \
            --serialization_dir tokenizers/bert_tokenizer
            --vocab_size 10000
```

and then using that tokenizer during preprocessing:

```
python -m scripts.preprocess_data \
            --train-path examples/ag/train.jsonl \
            --dev-path examples/ag/dev.jsonl \
            --tokenize \
            --tokenizer tokenizers/bert_tokenizer
            --serialization-dir examples/ag
```

This script will tokenize your data, and save the resulting output into the specified `serialization-dir`.

Alternatively, under `https://s3-us-west-2.amazonaws.com/allennlp/datasets/ag-news/preprocessed.tar", we have a tar file containing a pre-processed AG news data (with vocab size set to 30K). 

Run 

```
curl -Lo examples/ag/ag.tar https://s3-us-west-2.amazonaws.com/allennlp/datasets/ag-news/vampire_preprocessed_example.tar
tar -xvf examples/ag/ag.tar -C examples/
``` 

to access its contents.

In `examples/ag` (after running the `preprocess_data` module or unpacking `ag.tar`), you should see:

* `train.npz` - pre-computed bag of word representations of the training data
* `dev.npz` - pre-computed bag of word representations of the dev data
* `vampire.bgfreq` - background word frequencies
* `vocabulary/` - AllenNLP vocabulary directory

This script also creates a reference corpus to calcuate NPMI (normalized pointwise mutual information), a measure of topical coherence that we use for early stopping. By default, we use the validation data as our reference corpus. You can supply a `--reference-corpus-path` to the preprocessing script to use your own reference corpus.

In `examples/ag/reference`, you should see:

* `ref.npz` - pre-computed bag of word representations of the reference corpus (the dev data)
* `ref.vocab.json` - the reference corpus vocabulary


## Preprocessing large datasets

When preprocessing large datasets, it can be helpful to shard the output:

```
python -m scripts.preprocess_data \
            --train-path examples/ag/train.jsonl \
            --dev-path examples/ag/dev.jsonl \
            --tokenize \
            --vocab-size 30000 \
            --serialization-dir examples/ag
            --shard 10
```

This will make a 10-file shard of preprocessed training data in `examples/ag/shard`

We can then train on the shards using multiprocess VAMPIRE (see next section).

## Pretrain VAMPIRE

Set your data directory and vocabulary size as environment variables:

```

export DATA_DIR="$(pwd)/examples/ag"
export VOCAB_SIZE=30000 # This value is printed during the "preprocess_data" script.
```

If you're training on a dataset that's to large to fit into RAM, run VAMPIRE in lazy mode by additionally exporting:

```
export LAZY=1
```

Then train VAMPIRE:

```
python -m scripts.train \
            --config training_config/vampire.jsonnet \
            --serialization-dir model_logs/vampire \
            --environment VAMPIRE \
            --device -1 -o
```

This model can be run on a CPU (`--device -1`). To run on a GPU instead, run with `--device 0` (or any other available CUDA device number).

This command will output training logs at `model_logs/vampire`.

For convenience, we include the `--override` flag to remove the previous experiment at the same serialization directory.

## Multiprocess VAMPIRE

To train on a folder of training data shards, use multiprocess VAMPIRE:

```
python -m scripts.train \
            --config training_config/multiprocess_vampire.jsonnet \
            --serialization-dir model_logs/multiprocess_vampire \
            --environment MULTIPROCESS_VAMPIRE \
            --device -1 -o
```

## Inspect topics learned

During training, we output the learned topics after each epoch in the serialization directory, under `model_logs/vampire`.

After your model is finished training, check out the `best_epoch` field in `model_logs/vampire/metrics.json`, which corresponds to the training epoch at which NPMI is highest.

Then open up the corresponding epoch's file in `model_logs/vampire/topics/`.

## Use VAMPIRE with a downstream classifier

Using VAMPIRE with a downstream classifier is essentially the same as using regular ELMo. See [this documentation](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md#using-elmo-with-existing-allennlp-models) for details on how to do that.

This library has some convenience functions for including VAMPIRE with a downstream classifier. 

First, set some environment variables:
* `VAMPIRE_DIR`: path to newly trained VAMPIRE
* `VAMPIRE_DIM`: dimensionality of the newly trained VAMPIRE (the token embedder needs it explicitly)
* `THROTTLE`: the sample size of the data we want to train on.
* `EVALUATE_ON_TEST`: whether or not you would like to evaluate on test


```
export VAMPIRE_DIR="$(pwd)/model_logs/vampire"
export VAMPIRE_DIM=81
export THROTTLE=200
export EVALUATE_ON_TEST=0
```

Then, you can run the classifier:

```
python -m scripts.train \
            --config training_config/classifier.jsonnet \
            --serialization-dir model_logs/clf \
            --environment CLASSIFIER \
            --device -1 -o
```


As with VAMPIRE, this model can be run on a CPU (`--device -1`). To run on a GPU instead, run with `--device 0` (or any other available CUDA device number)

This command will output training logs at `model_logs/clf`.

The dataset sample (specified by `THROTTLE`) is governed by the global seed supplied to the trainer; the same seed will result in the same subsampling of training data. You can set an explicit seed by passing the additional flag `--seed` to the `train` module.

With 200 examples, we report a test accuracy of `83.9 +- 0.9` over 5 random seeds on the AG dataset. Note that your results may vary beyond these bounds under the low-resource setting.


## Using VAMPIRE as a Predictor

To generate VAMPIRE embeddings for a dataset, you can use VAMPIRE as a predictor.

First, add an index to the training data:

```
jq -rc '. + {"index": input_line_number}' examples/ag/train.jsonl > examples/ag/train.index.jsonl

```

Then, shard the input file, choosing the number of lines in each shard based on size of overall dataset:

```
mkdir  $(pwd)/examples/ag/shards/
split --lines 10000 --numeric-suffixes examples/ag/train.index.jsonl examples/ag/shards/
```

Then, run VAMPIRE in parallel on the data, using GNU parallel:

```
mkdir  $(pwd)/examples/ag/vampire_embeddings
parallel --ungroup python -m scripts.predict $(pwd)/model_logs/vampire/model.tar.gz {1} \
         --batch 64 \
         --include-package vampire \
         --predictor vampire \
         --output-file $(pwd)/examples/ag/vampire_embeddings/{1/.} \
         --silent :::  $(pwd)/examples/ag/shards/*
```

This will generate embeddings corresponding to each shard in `$(pwd)/examples/ag/vampire_embeddings`. 

Each file is a pytorch matrix serialization, containing a tuple of ids and embeddings:

```
>>> import torch
>>> len(torch.load('examples/ag/vampire_embeddings/00'))
2
>>> torch.load('examples/ag/vampire_embeddings/00')[0].shape ## indices
torch.Size([10000, 1])
>>> torch.load('examples/ag/vampire_embeddings/00')[1].shape ## vampire embeddings
torch.Size([10000, 81])
```

Each id corresponds to an ID in the `examples/ag/train.index.jsonl`, and is aligned with the associated vampire embedding.
