local NUM_GPUS = 1;
local BASE_READER = import "base_reader.jsonnet";
// throttle training data
local THROTTLE = 100;
local SEED = 5;

{
    "random_seed": SEED,
    "numpy_seed": SEED,
    "torch_seed": SEED,
    "dataset_reader": BASE_READER  + if std.type(THROTTLE) != "null" then { "sample": THROTTLE },
    "validation_dataset_reader": BASE_READER,
    "vocabulary":{
        "type": "bg_dumper",
    },
  "datasets_for_vocab_creation": ["train"],
  "train_data_path": "/home/ubuntu/vae/dump/imdb/train.jsonl",
  "validation_data_path": "/home/ubuntu/vae/dump/imdb/test.jsonl",
    "model": {
        "type": "nvdm",
        "track_topics": true,
        "topic_log_interval": 200,
        "update_background_freq": true,
        "distribution": {
            "type": "logistic_normal",
            "apply_batchnorm": true,
            "theta_dropout": 0.2,
            "theta_softmax": true
        },
        "latent_dim": 128,
        "kl_weight_annealing": "sigmoid",
        "dropout": 0.2,
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "bag_of_word_counts",
                    "vocab_namespace": "vae"
                }
            }
        },
        "encoder": {
            "type": "bow",
            "hidden_dim": 512
        },
        "decoder": {
            "type": "bow",
            "apply_batchnorm": false
        }
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 32
    },
     "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.002
        },
        "validation_metric": "-nll",
        "num_serialized_models_to_keep": 2,
        "num_epochs": 100,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": if NUM_GPUS > 1 then std.range(0, NUM_GPUS - 1) else 0,
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 0
        }
    }
}

