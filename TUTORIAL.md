# VAMPIRE Tutorial

This tutorial outlines the general process of training and using VAMPIRE for text classification. We will use the `AG News` corpus for all examples, but you can substitute any dataset you'd like.

## Quick links

* [Download data](#Download-Data)
* [Tokenization](#Pretokenize-Data)
* [Preprocessing](#Preprocess-Data)
* [Descripton of preprocessing output](#Description-Of-Preprocessing-Output)
* [Pretraining VAMPIRE](#Pretrain-VAMPIRE)
* [Inspecting learned topics](#Inspect-Learned-Topics)
* [Using VAMPIRE embeddings for classification](#Use-VAMPIRE-With-a-Downstream-Classifier)
* [Compute VAMPIRE embeddings for external use](#Compute-VAMPIRE-embeddings-for-External-Use)
## Download Data

Download the `AG News` corpus using the following script:

```bash
sh scripts/download_ag.sh
```

This will make an `examples/ag` directory with train, dev, test files from the `AG News` corpus.

This library expects our data to be in json format, where each line of each file corresponds to a separate instance. Each line must contain a `text` field, and optionally a `label` field. 

### (Optional) Download preprocessed data

Alternatively, you can download a tar file containing a pre-processed AG news data (with vocab size set to 30K). 

```bash
curl -Lo examples/ag/ag.tar \
    https://s3-us-west-2.amazonaws.com/allennlp/datasets/ag-news/vampire_preprocessed_example.tar
tar -xvf examples/ag/ag.tar -C examples/
``` 


## Tokenize Data

First, tokenize the data. If your data is already tokenized, you can copy your files into a new directory `examples/ag/tokenized`, and move on to the next step.

You can use spacy:

```bash
mkdir examples/ag/tokenized
cat examples/ag/train.jsonl | python -m scripts.pretokenize --tokenizer spacy \
                                                            --json \
                                                            --lower > examples/ag/tokenized/train.jsonl
cat examples/ag/dev.jsonl | python -m scripts.pretokenize --tokenizer spacy \
                                                          --json \
                                                          --lower > examples/ag/tokenized/dev.jsonl
cat examples/ag/test.jsonl | python -m scripts.pretokenize --tokenizer spacy \
                                                          --json \
                                                          --lower > examples/ag/tokenized/test.jsonl
```

or you can train a BPE, byte-level BPE (BBPE), or BERT tokenizer and then tokenize:

```bash
# use the jq library to get the raw training text
jq -r '.text' examples/ag/train.jsonl > examples/ag/train.txt
python -m scripts.train_tokenizer --input_file examples/ag/train.txt \
            --tokenizer_type BERT \
            --serialization_dir tokenizers/bert_tokenizer \
            --vocab_size 10000
mkdir examples/ag/tokenized
cat examples/ag/train.jsonl | python -m scripts.pretokenize --tokenizer tokenizers/bert_tokenizer \
                                                            --json \
                                                            --lower > examples/ag/tokenized/train.jsonl
cat examples/ag/dev.jsonl | python -m scripts.pretokenize --tokenizer tokenizers/bert_tokenizer \
                                                            --json \
                                                            --lower > examples/ag/tokenized/dev.jsonl
cat examples/ag/test.jsonl | python -m scripts.pretokenize --tokenizer tokenizers/bert_tokenizer \
                                                            --json \
                                                            --lower > examples/ag/tokenized/test.jsonl
```

## Preprocess data

To make pretraining fast, we next precompute fixed bag-of-words representations of the data using pre-tokenized data.

```bash
python -m scripts.preprocess_data \
            --train-path examples/ag/tokenized/train.jsonl \
            --dev-path examples/ag/tokenized/dev.jsonl \
            --serialization-dir examples/ag
```

This script will tokenize your data, and save the resulting output into the specified `serialization-dir`.



## Description of Preprocessing Output

In `examples/ag` (after running the `preprocess_data` module or unpacking `ag.tar`), you should see:

* `train.npz` - pre-computed bag of word representations of the training data
* `dev.npz` - pre-computed bag of word representations of the dev data
* `vampire.bgfreq` - background word frequencies
* `vocabulary/` - AllenNLP vocabulary directory

This script also creates a reference corpus to calcuate NPMI (normalized pointwise mutual information), a measure of topical coherence that we use for early stopping. By default, we use the validation data as our reference corpus. You can supply a `--reference-corpus-path` to the preprocessing script to use your own reference corpus.

In `examples/ag/reference`, you should see:

* `ref.npz` - pre-computed bag of word representations of the reference corpus (the dev data)
* `ref.vocab.json` - the reference corpus vocabulary


## Pretrain VAMPIRE

Set your data directory and vocabulary size as environment variables:

```bash
export DATA_DIR="$(pwd)/examples/ag"
export VOCAB_SIZE=3960 # This value is printed during the "preprocess_data" script.
```

If you're training on a dataset that's to large to fit into RAM, run VAMPIRE in lazy mode by additionally exporting:

```bash
export LAZY=1
```

Then train VAMPIRE:

```bash
python -m scripts.train \
            --config training_config/vampire.jsonnet \
            --serialization-dir model_logs/vampire \
            --environment VAMPIRE \
            --device 0 -o
```

This model can be run on a CPU (`--device -1`). To run on a GPU instead, run with `--device 0` (or any other available CUDA device number).

This command will output training logs at `model_logs/vampire`.

For convenience, we include the `--override` flag to remove the previous experiment at the same serialization directory.

## Inspect Learned Topics

During training, we output the learned topics after each epoch in the serialization directory, under `model_logs/vampire`.

After your model is finished training, check out the `best_epoch` field in `model_logs/vampire/metrics.json`, which corresponds to the training epoch at which NPMI is highest.

Then open up the corresponding epoch's file in `model_logs/vampire/topics/`.



## Use VAMPIRE With a Downstream Classifier

Using VAMPIRE with a downstream classifier is essentially the same as using regular ELMo. See [this documentation](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md#using-elmo-with-existing-allennlp-models) for details on how to do that.

This library has some convenience functions for including VAMPIRE with a downstream classifier. 

First, set some environment variables:
* `VAMPIRE_DIR`: path to newly trained VAMPIRE
* `VAMPIRE_DIM`: dimensionality of the newly trained VAMPIRE (the token embedder needs it explicitly)
* `THROTTLE`: the sample size of the data we want to train on.
* `EVALUATE_ON_TEST`: whether or not you would like to evaluate on test


```bash
export VAMPIRE_DIR="$(pwd)/model_logs/vampire"
export VAMPIRE_DIM=81
export THROTTLE=200
export EVALUATE_ON_TEST=0
```

Then, you can run the classifier:

```bash
srun -w allennlp-server3 --gpus=1 -p allennlp_hipri python -m scripts.train \
            --config training_config/classifier.jsonnet \
            --serialization-dir model_logs/clf \
            --environment CLASSIFIER \
            --device 0 -o
```


As with VAMPIRE, this model can be run on a CPU (`--device -1`). To run on a GPU instead, run with `--device 0` (or any other available CUDA device number)

This command will output training logs at `model_logs/clf`.

The dataset sample (specified by `THROTTLE`) is governed by the global seed supplied to the trainer; the same seed will result in the same subsampling of training data. You can set an explicit seed by passing the additional flag `--seed` to the `train` module.

With 200 examples, we report a test accuracy of `83.9 +- 0.9` over 5 random seeds on the AG dataset. Note that your results may vary beyond these bounds under the low-resource setting.

## Compute VAMPIRE Embeddings for External Use

To generate VAMPIRE embeddings for a dataset, you can use VAMPIRE as a predictor. 

First, add an index to the training data, using the [jq](https://stedolan.github.io/jq/) library:

```bash
jq -rc '. + {"index": input_line_number}' examples/ag/tokenized/dev.jsonl \
        > examples/ag/tokenized/dev.index.jsonl
```

Then, run VAMPIRE on the data:

```
python -m scripts.predict $(pwd)/model_logs/vampire/model.tar.gz  \
         examples/ag/tokenized/dev.index.jsonl \
         --batch 64 \
         --include-package vampire \
         --predictor vampire \
         --output-file $(pwd)/examples/ag/dev_embeddings.pt
         --silent
```


The output file in `$(pwd)/examples/ag/vampire_embeddings.pt` is a pytorch matrix serialization, containing a tuple of ids and embeddings:

```python
>>> import torch
>>> len(torch.load('examples/ag/dev_embeddings.pt'))
2
>>> torch.load('examples/ag/dev_embeddings.pt')[0].shape ## indices
torch.Size([5000, 1])
>>> torch.load('examples/ag/dev_embeddings.pt')[1].shape ## vampire embeddings of dimension 81
torch.Size([5000, 81])
```

Each id corresponds to an ID in the `examples/ag/dev.index.jsonl`, and is aligned with the associated vampire embedding.


## What's next?

You can read about how to [scale up VAMPIRE](SCALING.md) for large corpora, or learn about how to [troubleshoot VAMPIRE]((TROUBLESHOOTING.md) if you run into issues.