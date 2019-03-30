local VAE_FIELDS(MODEL_ARCHIVE, BG_FREQ, VOCAB_NAME) = {
    "vae_embedder": {
        VOCAB_NAME: {
                "type": "vae_token_embedder",
                // "scalar_mix": [
                //     std.parseInt(std.extVar("L1")), std.parseInt(std.extVar("L2")), std.parseInt(std.extVar("L3"))
                // ],
                "expand_dim": false,
                "requires_grad": false,
                "model_archive": MODEL_ARCHIVE,
                "background_frequency": BG_FREQ,
                "dropout": 5 / 10.0
        }
    }
};

{
    "dataset_reader": {
        "type": "semisupervised_text_classification_json",
        "ignore_labels": true,
        "lazy": false,
        "sample": null,
        "max_sequence_length": 400,
        "token_indexers": {
            "vae_tokens_1": {
                "type": "single_id",
                "lowercase_tokens": true,
                "namespace": "vae_tokens_1"
            },
            "vae_tokens_2": {
                "type": "single_id",
                "lowercase_tokens": true,
                "namespace": "vae_tokens_2"
            },
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
    },
    "iterator": {
        "type": "basic",
        "batch_size": 128,
        "track_epoch": true
    },
    "model": {
        "type": "stacked_unsupervised",
        "apply_batchnorm": true,
        "vae_embedder_1": {
            "token_embedders": {
                "vae_tokens_1": {
                "type": "vae_token_embedder",
                // "scalar_mix": [
                //     std.parseInt(std.extVar("L1")), std.parseInt(std.extVar("L2")), std.parseInt(std.extVar("L3"))
                // ],
                "expand_dim": false,
                "requires_grad": false,
                "model_archive": "s3://suching-dev/pretrained-models/vampire/imdb/small_10K/model.tar.gz",
                "background_frequency": "s3://suching-dev/pretrained-models/vampire/imdb/small_10K/vocabulary/vae.bgfreq.json",
                "dropout": 0
                }
            },
            "allow_unmatched_keys": true
        },
        "vae_embedder_2": {
            "token_embedders": {
                "vae_tokens_2": {
                "type": "vae_token_embedder",
                // "scalar_mix": [
                //     std.parseInt(std.extVar("L1")), std.parseInt(std.extVar("L2")), std.parseInt(std.extVar("L3"))
                // ],
                "expand_dim": false,
                "requires_grad": false,
                "model_archive": "s3://suching-dev/pretrained-models/vampire/ag-news/small_10K/model.tar.gz",
                "background_frequency": "s3://suching-dev/pretrained-models/vampire/ag-news/small_10K/vocabulary/vae.bgfreq.json",
                "dropout": 0
                }
            },
            "allow_unmatched_keys": true
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
                    128
                ],
                "input_dim": 128,
                "num_layers": 1
            },
            "encoder": {
                "activations": [
                    "softplus",
                    "softplus"
                ],
                "hidden_dims": [
                    128,
                    128
                ],
                "input_dim": 128,
                "num_layers": 2
            },
            "log_variance_projection": {
                "activations": [
                    "linear"
                ],
                "hidden_dims": [
                    128
                ],
                "input_dim": 128,
                "num_layers": 1
            },
            "mean_projection": {
                "activations": [
                    "linear"
                ],
                "hidden_dims": [
                    128
                ],
                "input_dim": 128,
                "num_layers": 1
            },
            "z_dropout": 0.2
        },
    //     "initializer": [ 
    //         [
    //         "vae.mean_projection._linear_layers.0.weight|vae._decoder.weight|vae.encoder._linear_layers.0.weight|vae.encoder._linear_layers.0.bias|vae.encoder._linear_layers.1.weight|vae.encoder._linear_layers.1.bias|vae.log_variance_projection._linear_layers.0.weight|vae.log_variance_projection._linear_layers.0.bias|vae.mean_projection._linear_layers.0.bias", 
    //         {
    //             "type": "pretrained",
    //             "weights_file_path": "/Users/suching/Github/vae/weights.th",
    //             "parameter_name_overrides": {
    //                 "vae.encoder._linear_layers.0.weight": "vae.encoder._linear_layers.0.weight",
    //                 "vae.encoder._linear_layers.0.bias": "vae.encoder._linear_layers.0.bias",
    //                 "vae._decoder.weight": "vae._decoder.weight",
    //                 "vae.encoder._linear_layers.1.weight": "vae.encoder._linear_layers.1.weight",
    //                 "vae.encoder._linear_layers.1.bias": "vae.encoder._linear_layers.1.bias",
    //                 "vae.mean_projection._linear_layers.0.weight": "vae.mean_projection._linear_layers.0.weight",
    //                 "vae.mean_projection._linear_layers.0.bias":  "vae.mean_projection._linear_layers.0.bias",
    //                 "vae.log_variance_projection._linear_layers.0.weight": "vae.log_variance_projection._linear_layers.0.weight",
    //                 "vae.log_variance_projection._linear_layers.0.bias": "vae.log_variance_projection._linear_layers.0.bias"
    //             }
    //         }
    //     ]
    // ]
    },
    "train_data_path": "s3://suching-dev/final-datasets/imdb/train_pretokenized.jsonl",
    "validation_data_path": "s3://suching-dev/final-datasets/imdb/dev_pretokenized.jsonl",
    "trainer": {
        "cuda_device": -1,
        "num_epochs": 50,
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "patience": 5,
        "validation_metric": "-nll"
    },
    "vocabulary": {
        "type": "vocabulary_with_two_vaes",
        "vae_vocab_file_1": "s3://suching-dev/pretrained-models/vampire/imdb/small_10K/vocabulary/vae.txt",
        "vae_vocab_file_2": "s3://suching-dev/pretrained-models/vampire/ag-news/small_10K/vocabulary/vae.txt",
    },
    "validation_dataset_reader": {
        "type": "semisupervised_text_classification_json",
        "ignore_labels": true,
        "lazy": false,
        "sample": null,
        "max_sequence_length": 400,
        "token_indexers": {
            "vae_tokens_1": {
                "type": "single_id",
                "lowercase_tokens": true,
                "namespace": "vae_tokens_1"
            },
            "vae_tokens_2": {
                "type": "single_id",
                "lowercase_tokens": true,
                "namespace": "vae_tokens_2"
            },
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
    }
}