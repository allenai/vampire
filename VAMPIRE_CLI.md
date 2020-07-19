
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
