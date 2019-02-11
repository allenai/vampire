local CUDA_DEVICE =
  if std.parseInt(std.extVar("NUM_GPU")) == 0 then
    -1
  else if std.parseInt(std.extVar("NUM_GPU")) > 1 then
    std.range(0, std.extVar("NUM_GPU") - 1)
  else if std.parseInt(std.extVar("NUM_GPU")) == 1 then
    0;


{
    "numpy_seed": 16011988,
    "random_seed": 16011988,
    "pytorch_seed": 16011988,
    "dataset_reader": {
        "type": "semisupervised_text_classification_json",
        "lazy": false,
        "sample": null,
        "sequence_length": 400,
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true,
                "namespace": "classifier"
            }
        },
        "tokenizer": {
            "word_splitter": "spacy"
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size": 128
    },
    "model": {
        "type": "classifier",
        "dropout": 0.5,
        "input_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "bag_of_word_counts",
                    "ignore_oov": "true",
                    "vocab_namespace": "classifier"
                }
            }
        }
    },
    "train_data_path": "s3://suching-dev/imdb/train.jsonl",
    "validation_data_path": "s3://suching-dev/imdb/dev.jsonl",
    "trainer": {
        "cuda_device": CUDA_DEVICE,
        "num_epochs": 200,
        "optimizer": {
            "type": "adam",
            "lr": 0.0001
        },
        "patience": 20,
        "validation_metric": "+accuracy"
    },
    "datasets_for_vocab_creation": [
        "train"
    ],
    "validation_dataset_reader": {
        "type": "semisupervised_text_classification_json",
        "lazy": false,
        "sample": null,
        "sequence_length": 400,
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true,
                "namespace": "classifier"
            }
        },
        "tokenizer": {
            "word_splitter": "spacy"
        }
    }
}