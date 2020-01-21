# Scaling VAMPIRE to Large Datasets

The [basic tutorial](TUTORIAL.md) describes training VAMPIRE on a smaller dataset called `AG News`. The advantage of using VAMPIRE is in its lightweightness and speed, which means one can train on much larger datasets in a reasonable amount of time.

This document details some alternative commands one can use when training on larger corpora. Most of these commands involve using [pigz](https://zlib.net/pigz/), [jq](https://stedolan.github.io/jq/), and [GNU parallel](https://www.gnu.org/software/parallel/), so please install those on your machine.

## Download a large dataset

In this tutorial, we will train VAMPIRE on the [1B words corpus](https://www.statmt.org/lm-benchmark/).

To download the corpus, run the following command:

```bash
curl -Lo 1b.corpus.gz https://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
```

## Pretokenizing large datasets

You can use `pigz` and GNU parallel with this library to tokenize large datasets, leveraging all CPUs in your machine.

First, take a sample of the corpus to train the tokenizer. Here we use `awk` to sample 25% of the corpus, which will end up around 1GB.

```bash
zcat 1b.corpus.tar.gz | parallel --pipe -q awk 'BEGIN {srand()} !/^$/ { if (rand() <= .25) print $0}' | sed '/^[[:space:]]*$/d' > 1b.sample
```

Then train the tokenizer on that sample

```bash
python -m scripts.train_tokenizer \
            --input_file 1b.sample \
            --tokenizer_type BERT \
            --serialization_dir tokenizers/bert_tokenizer \
            --vocab_size 10000
```

Finally, use GNU parallel to speed up tokenization with your pretrained tokenizer

```
zcat 1b.corpus.gz | parallel --pipe -q python -m scripts.pretokenize --tokenizer tokenizers/bert_tokenizer --lower --silent  > 1b.corpus.tokenized.txt
```

On Linux machines, use the application `pv` to track how fast data is being tokenized:
```
zcat 1b.corpus.gz | pv | parallel --pipe -q python -m scripts.pretokenize --tokenizer tokenizers/bert_tokenizer --lower --silent  > 1b.corpus.tokenized.txt
```

## Preprocessing large datasets

First, make a dev set from the training data:

```
zcat 1b.corpus.tar.gz | parallel --pipe -q awk 'BEGIN {srand()} !/^$/ { if (rand() <= .01) print $0}' | sed '/^[[:space:]]*$/d' > 1b.sample
```

When preprocessing large datasets, it can be helpful to shard the output:

```bash
python -m scripts.preprocess_data \
            --train-path 1b.corpus.tokenized.txt \
            --serialization-dir 1b \
            --shard 100
```

This will make a 100-file shard of preprocessed training data in `examples/ag/preprocessed_shards`

We can then train on the shards using multiprocess VAMPIRE (see next section).


## Training Multiprocess VAMPIRE

Set your data directory and vocabulary size as environment variables:

```bash
export DATA_DIR="$(pwd)/1b/"
export VOCAB_SIZE=8335 # This value is printed during the "preprocess_data" script.
```

If you're training on dataset shards that are too large to fit into RAM, run VAMPIRE in lazy mode by additionally exporting:

```bash
export LAZY=1
```

To train on a folder of training data shards, use multiprocess VAMPIRE:

```bash
srun -w allennlp-server3 --gpus=1 -p allennlp_hipri python -m scripts.train \
            --config training_config/multiprocess_vampire.jsonnet \
            --serialization-dir model_logs/multiprocess_vampire \
            --environment MULTIPROCESS_VAMPIRE \
            --device 0 -o
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

