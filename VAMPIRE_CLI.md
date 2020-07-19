
## Preprocess Data

```
python -m scripts.preprocess_data \
            --train-path examples/ag/train.jsonl \
            --dev-path examples/ag/dev.jsonl \
            --tokenize \
            --tokenizer-type spacy \
            --vocab-size 30000 \
            --serialization-dir examples/ag
```

This script will tokenize your data, and save the resulting output into the specified `serialization-dir`.

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

```
export DATA_DIR="$(pwd)/examples/ag"
export VOCAB_SIZE=30000
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
            --device -1
```

This model can be run on a CPU (`--device -1`). To run on a GPU instead, run with `--device 0` (or any other available CUDA device number).

This command will output training logs at `model_logs/vampire`.

For convenience, we include the `--override` flag to remove the previous experiment at the same serialization directory.
