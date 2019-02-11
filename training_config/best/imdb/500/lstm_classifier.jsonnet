local CUDA_DEVICE =
  if std.parseInt(std.extVar("NUM_GPU")) == 0 then
    -1
  else if std.parseInt(std.extVar("NUM_GPU")) > 1 then
    std.range(0, std.extVar("NUM_GPU") - 1)
  else if std.parseInt(std.extVar("NUM_GPU")) == 1 then
    0;

{
    "dataset_reader": {
        "type": "semisupervised_text_classification_json",
        "lazy": false,
        "sample": "500",
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
        "dropout": 0.2,
        "encoder": {
            "type": "seq2seq",
            "aggregations": "meanpool",
            "architecture": {
                "type": "lstm",
                "bidirectional": true,
                "hidden_size": 64,
                "input_size": 300,
                "num_layers": 2
            }
        },
        "input_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 300,
                    "trainable": true,
                    "vocab_namespace": "classifier"
                }
            }
        }
    },
    "train_data_path": "s3://suching-dev/imdb/train.jsonl",
    "validation_data_path": "s3://suching-dev/imdb/dev.jsonl",
    "trainer": {
        "cuda_device": 0,
        "num_epochs": 200,
        "optimizer": {
            "type": "adam",
            "lr": 0.001
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