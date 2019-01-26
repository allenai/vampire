local VAE_FIELDS = import 'vae_fields.jsonnet';

local NUM_GPUS = 1;
// throttle training data
local THROTTLE = 100;
local SEED = 5;
// add vae embeddings
local ADD_VAE = true;

local VAE_FIELDS = {
    "vae_indexer": {
        "vae_tokens": {
            "type": "single_id",
            "namespace": "vae",
            "lowercase_tokens": true
        }
    },  
    "vae_embedder": {
        "vae_tokens": {
            "type": "vae_token_embedder",
            "representation": "encoder_output",
            "expand_dim": false,
            "model_archive": "/home/ubuntu/vae/model_logs/nvdm/model.tar.gz",
            "combine": false, 
            "dropout": 0.2
        }
    }
};

local BASE_READER(add_vae, throttle) = {
        "lazy": false,
        "type": "semisupervised_text_classification_json",
        "tokenizer": {
            "word_splitter": "spacy",
            "word_filter": {
                "type": "regex_and_stopwords",
                "patterns": [
                            // "\\w{1,3}\\b", // tokens of length <= 3
                            //  "\\w*\\d+\\w*", // words that contain digits,
                             "\\w*[^\\P{P}\\-]+\\w*" // punctuation
                            ],
                "stopword_file": "/home/ubuntu/vae/vae/common/stopwords/snowball_stopwords.txt"
            }
        },
        "token_indexers": {
            "tokens": {
                    "type": "single_id",
                    "namespace": "tokens",
                    "lowercase_tokens": true
                }
        } + if add_vae then VAE_FIELDS['vae_indexer'] else {}, 
        "ignore_labels": true,
        "shift_target": false,
        "sequence_length": 400,
        if std.type(THROTTLE) != "null" then { "sample": THROTTLE }
};

local EMBEDDER(add_vae) = {
            "token_embedders": {
                "tokens": {
                    "type": "bag_of_word_counts",
                }
            } + if add_vae then VAE_FIELDS['vae_embedder'] else {}
};

{
    "random_seed": SEED,
    "numpy_seed": SEED,
    "torch_seed": SEED,
    "dataset_reader": BASE_READER(ADD_VAE, THROTTLE),
    "validation_dataset_reader": BASE_READER(ADD_VAE, null),
    "datasets_for_vocab_creation": ["train"],
    "train_data_path": "/home/ubuntu/vae/dump/imdb/train.jsonl",
    "validation_data_path": "/home/ubuntu/vae/dump/imdb/test.jsonl",
    "model": {
        "type": "logistic_regression",
        "text_field_embedder": EMBEDDER(ADD_VAE)
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 32
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 2,
        "num_epochs": 75,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": 0,
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 0
        }
    }
}

