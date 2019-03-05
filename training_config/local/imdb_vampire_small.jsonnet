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
                    "<",
                    "$",
                    "href=",
                    "|",
                    "\u00b0",
                    "+",
                    "\u00a3"
                ]
            },
            "word_splitter": "just_spaces"
        },
        "unlabeled_data_path": "s3://suching-dev/final-datasets/imdb/unlabeled_pretokenized.jsonl"
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
        "reference_counts": "s3://suching-dev/final-datasets/imdb/valid_npmi_reference/train.npz",
        "reference_vocabulary": "s3://suching-dev/final-datasets/imdb/valid_npmi_reference/train.vocab.json",
        "update_background_freq": false,
        "vae": {
            "type": "logistic_normal",
            "apply_batchnorm": false,
            "decoder": {
                "activations": [
                    "tanh"
                ],
                "hidden_dims": [
                    10002
                ],
                "input_dim": 64,
                "num_layers": 1
            },
            "encoder": {
                "activations": [
                    "softplus",
                    "softplus"
                ],
                "hidden_dims": [
                    64,
                    64
                ],
                "input_dim": 10002,
                "num_layers": 2
            },
            "log_variance_projection": {
                "activations": [
                    "linear"
                ],
                "hidden_dims": [
                    64
                ],
                "input_dim": 64,
                "num_layers": 1
            },
            "mean_projection": {
                "activations": [
                    "linear"
                ],
                "hidden_dims": [
                    64
                ],
                "input_dim": "64",
                "num_layers": 1
            },
            "z_dropout": 0.2
        }
    },
    "train_data_path": "s3://suching-dev/final-datasets/imdb/train_pretokenized.jsonl",
    "validation_data_path": "s3://suching-dev/final-datasets/imdb/dev_pretokenized.jsonl",
    "trainer": {
        "cuda_device": 0,
        "num_epochs": 50,
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "patience": 5,
        "validation_metric": "+npmi"
    },
    "vocabulary": {
        "type": "bg_dumper",
        "max_vocab_size": {
            "vae": 10000
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
                    "<",
                    "$",
                    "href=",
                    "|",
                    "\u00b0",
                    "+",
                    "\u00a3"
                ]
            },
            "word_splitter": "just_spaces"
        },
        "unlabeled_data_path": null
    }
}