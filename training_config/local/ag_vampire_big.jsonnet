{
    "dataset_reader": {
        "type": "semisupervised_text_classification_json",
        "ignore_labels": true,
        "lazy": false,
        "sample": null,
        "sequence_length": 400,
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true,
                "namespace": "vae"
            }
        },
        "tokenizer": {
            "word_filter": {
                "type": "regex_and_stopwords",
                "patterns": [
                    "\\w{1,3}\\b",
                    "\\w*\\d+\\w*",
                    "\\w*[^\\P{P}]+\\w*"
                ],
                "stopword_file": "s3://suching-dev/stopwords/snowball_stopwords.txt",
                "tokens_to_add": [
                    ">",
                    "<"
                ]
            },
            "word_splitter": "spacy"
        },
        "unlabeled_data_path": null
    },
    "iterator": {
        "type": "basic",
        "batch_size": 128,
        "track_epoch": true
    },
    "model": {
        "type": "nvdm",
        "apply_batchnorm": true,
        "bow_embedder": {
            "type": "bag_of_word_counts",
            "ignore_oov": true,
            "vocab_namespace": "vae"
        },
        "kl_weight_annealing": "sigmoid",
        "reference_counts": "s3://suching-dev/final-datasets/ag-news/valid_npmi_reference/train.npz",
        "reference_vocabulary": "s3://suching-dev/final-datasets/ag-news/valid_npmi_reference/train.vocab.json",
        "update_background_freq": false,
        "vae": {
            "type": "logistic_normal",
            "apply_batchnorm": false,
            "decoder": {
                "activations": [
                    "tanh"
                ],
                "hidden_dims": [
                    30002
                ],
                "input_dim": 512,
                "num_layers": 1
            },
            "encoder": {
                "activations": [
                    "softplus",
                    "softplus",
                    "softplus"
                ],
                "hidden_dims": [
                    512,
                    512,
                    512
                ],
                "input_dim": 30002,
                "num_layers": 3
            },
            "log_variance_projection": {
                "activations": [
                    "linear"
                ],
                "hidden_dims": [
                    512
                ],
                "input_dim": 512,
                "num_layers": 1
            },
            "mean_projection": {
                "activations": [
                    "linear"
                ],
                "hidden_dims": [
                    512
                ],
                "input_dim": "512",
                "num_layers": 1
            },
            "z_dropout": 0.5
        }
    },
    "train_data_path": "s3://suching-dev/final-datasets/ag-news/train.jsonl",
    "validation_data_path": "s3://suching-dev/final-datasets/ag-news/dev.jsonl",
    "trainer": {
        "cuda_device": -1,
        "num_epochs": 200,
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "patience": 75,
        "validation_metric": "+npmi"
    },
    "vocabulary": {
        "type": "bg_dumper",
        "max_vocab_size": {
            "vae": 30000
        }
    },
    "datasets_for_vocab_creation": [
        "train"
    ],
    "validation_dataset_reader": {
        "type": "semisupervised_text_classification_json",
        "ignore_labels": true,
        "lazy": false,
        "sample": null,
        "sequence_length": 400,
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true,
                "namespace": "vae"
            }
        },
        "tokenizer": {
            "word_filter": {
                "type": "regex_and_stopwords",
                "patterns": [
                    "\\w{1,3}\\b",
                    "\\w*\\d+\\w*",
                    "\\w*[^\\P{P}]+\\w*"
                ],
                "stopword_file": "s3://suching-dev/stopwords/snowball_stopwords.txt",
                "tokens_to_add": [
                    ">",
                    "<"
                ]
            },
            "word_splitter": "spacy"
        },
        "unlabeled_data_path": null
    }
}