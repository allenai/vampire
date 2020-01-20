# Scaling VAMPIRE to Large Datasets

The [basic tutorial](TUTORIAL.md) describes training VAMPIRE on a smaller dataset called `AG News`. The advantage of using VAMPIRE is in its lightweightness and speed, which means one can train on much larger datasets in a reasonable amount of time.

This document details some alternative commands one can use when training on larger corpora. Most of these commands involve using [jq](https://stedolan.github.io/jq/) and [GNU parallel](https://www.gnu.org/software/parallel/), so please install those on your machine.

## Pretokenizing large datasets

You can use GNU parallel with this library to tokenize large datasets, leveraging all CPUs in your machine.

```
cat examples/ag/train.jsonl | parallel --pipe -q jq -r '.text' > examples/ag/train.txt
python -m scripts.train_tokenizer --input_file examples/ag/train.txt \
            --tokenizer_type BERT \
            --serialization_dir tokenizers/bert_tokenizer \
            --vocab_size 10000
mkdir examples/ag/tokenized
cat examples/ag/train.jsonl | parallel --pipe -q python -m scripts.pretokenize --tokenizer tokenizers/bert_tokenizer \
--json \
--lower --silent > examples/ag/tokenized/train.jsonl
cat examples/ag/dev.jsonl | parallel --pipe -q python -m scripts.pretokenize --tokenizer tokenizers/bert_tokenizer \
--json \
--lower --silent > examples/ag/tokenized/dev.jsonl
cat examples/ag/test.jsonl | parallel --pipe -q python -m scripts.pretokenize --tokenizer tokenizers/bert_tokenizer \
--json \
--lower --silent > examples/ag/tokenized/test.jsonl
```

On Linux machines, use the application `pv` to track how fast data is being tokenized:
```
cat examples/ag/train.jsonl | pv | parallel --pipe -q python -m scripts.pretokenize --tokenizer tokenizers/bert_tokenizer --json --lower --silent > examples/ag/tokenized/train.jsonl
```

## Preprocessing large datasets

When preprocessing large datasets, it can be helpful to shard the output:

```bash
python -m scripts.preprocess_data \
            --train-path examples/ag/tokenized/train.jsonl \
            --dev-path examples/ag/tokenized/dev.jsonl \
            --serialization-dir examples/ag \
            --shard 10
```

This will make a 10-file shard of preprocessed training data in `examples/ag/preprocessed_shards`

We can then train on the shards using multiprocess VAMPIRE (see next section).


## Training Multiprocess VAMPIRE

To train on a folder of training data shards, use multiprocess VAMPIRE:

```bash
python -m scripts.train \
            --config training_config/multiprocess_vampire.jsonnet \
            --serialization-dir model_logs/multiprocess_vampire \
            --environment MULTIPROCESS_VAMPIRE \
            --device -1 -o
```

This will spawn workers to feed the different shards into VAMPIRE. You can set the number of workers in `environments/environments.py`, with the `NUM_WORKERS` hyperparameter.

## Computing VAMPIRE embeddings for large datasets


To compute VAMPIRE embeddings for a very large dataset, we first, add an index to the training data:

```bash
jq -rc '. + {"index": input_line_number}' examples/ag/tokenized/dev.jsonl \
        > examples/ag/tokenized/dev.index.jsonl

Then, we shard the input file, choosing the number of lines in each shard based on size of overall dataset:

```bash
mkdir  $(pwd)/examples/ag/shards/
split --lines 10000 --numeric-suffixes examples/ag/tokenized/train.index.jsonl examples/ag/shards/
```

Then, run VAMPIRE in parallel on the data, using [GNU parallel](https://www.gnu.org/software/parallel/):

```bash
mkdir  $(pwd)/examples/ag/vampire_embeddings
parallel --ungroup python -m scripts.predict $(pwd)/model_logs/vampire/model.tar.gz {1} \
         --batch 64 \
         --include-package vampire \
         --predictor vampire \
         --output-file $(pwd)/examples/ag/vampire_embeddings/{1/.} \
         --silent :::  $(pwd)/examples/ag/shards/*
```

This will generate embeddings corresponding to each shard in `$(pwd)/examples/ag/vampire_embeddings`. 

The output file in `$(pwd)/examples/ag/vampire_embeddings/$SHARD` is a pytorch matrix serialization, containing a tuple of ids and embeddings:

```python
>>> import torch
>>> len(torch.load('examples/ag/vampire_embeddings/00'))
2
>>> torch.load('examples/ag/vampire_embeddings/00')[0].shape ## indices
torch.Size([10000, 1])
>>> torch.load('examples/ag/vampire_embeddings/00')[1].shape ## vampire embeddings of dimension 81
torch.Size([10000, 81])
```

Each id corresponds to an ID in the `examples/ag/train.index.jsonl`, and is aligned with the associated vampire embedding.

