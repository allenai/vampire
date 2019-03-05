

{
    "dataset_reader": {
        "type": "semisupervised_text_classification_json",
        "lazy": false,
        "sample": "10000",
        "sequence_length": 400,
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true,
                "namespace": "classifier"
            },
            "vae_tokens": {
            "type": "single_id",
            "namespace": "vae",
            "lowercase_tokens": true
            }
        },
        "tokenizer": {
            "word_splitter": "spacy"
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size": 32
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
                },
                "vae_tokens": {
                    "type": "vae_token_embedder",
                    "expand_dim": false,
                    "requires_grad": false,
                    "model_archive": "s3://suching-dev/best-npmi-vae-IMDB-final/model.tar.gz",
                    "background_frequency": "s3://suching-dev/best-npmi-vae-IMDB-final/vae.bgfreq.json",
                    "dropout": 0.2
                }
            }
        }
    },
    "train_data_path": "s3://suching-dev/final-datasets/imdb/train.jsonl",
    "validation_data_path": "s3://suching-dev/final-datasets/imdb/dev.jsonl",
    "trainer": {
        "cuda_device": 0,
        "num_epochs": 200,
        "optimizer": {
            "type": "adam",
            "lr": 0.0005
        },
        "patience": 20,
        "validation_metric": "+accuracy"
    },
    "datasets_for_vocab_creation": [
        "train"
    ],

    "s3://suching-dev/best-npmi-vae-IMDB-final/vae.txt"

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