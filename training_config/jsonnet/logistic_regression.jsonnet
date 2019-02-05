local NUM_GPUS = 0;
// throttle training data
local THROTTLE = 10000;
local SEED = 87;
// add vae embeddings
local ADD_VAE = true;
local ADD_ELMO = false;
local TRAIN_PATH = "s3://suching-dev/imdb/train.jsonl";
local DEV_PATH = "s3://suching-dev/imdb/dev.jsonl";
local STOPWORDS_PATH = "s3://suching-dev/stopwords/snowball_stopwords.txt";
// set to false during debugging
local USE_SPACY_TOKENIZER = false;

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
                    "model_archive": "/Users/suching/Github/vae/model_logs/saved_model/model.tar.gz",
                    "background_frequency": "/Users/suching/Github/vae/model_logs/saved_model/vae.bgfreq.json",
                    "dropout": 0.2
        }
    }
};


local ELMO_FIELDS = {
    "elmo_indexer": {
        "elmo": {
		        "type": "elmo_characters",
		}
    },  
    "elmo_embedder": {
        "elmo": {
            "type": "elmo_token_embedder",
            "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
            "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
            "do_layer_norm": false,
            "dropout": 0.2
        }
    }
};

local BASE_READER(add_vae, add_elmo, throttle, use_spacy_tokenizer) = {
        "lazy": false,
        "type": "semisupervised_text_classification_json",
        "tokenizer": {
            "word_splitter": if use_spacy_tokenizer then "spacy" else "just_spaces",
            "word_filter": {
                "type": "regex_and_stopwords",
                "patterns": [
                            // "\\w{1,3}\\b", // tokens of length <= 3
                            //  "\\w*\\d+\\w*", // words that contain digits,
                             "\\w*[^\\P{P}\\-]+\\w*" // punctuation
                            ],
                "stopword_file": STOPWORDS_PATH
            }
        },
        "token_indexers": {
            "tokens": {
                    "type": "single_id",
                    "namespace": "tokens",
                    "lowercase_tokens": true
                }
        } + if add_vae then VAE_FIELDS['vae_indexer'] else {} 
            + if add_elmo then ELMO_FIELDS['elmo_indexer'] else {}, 
        "sequence_length": 400,
        "sample": throttle,
};

local EMBEDDER(add_vae, add_elmo) = {
            "token_embedders": {
                "tokens": {
                    "type": "bag_of_word_counts",
                    "ignore_oov": true
                }
            } + if add_vae then VAE_FIELDS['vae_embedder'] else {}
                + if add_elmo then ELMO_FIELDS['elmo_embedder'] else {}, 
};

{
    "random_seed": SEED,
    "numpy_seed": SEED,
    "pytorch_seed": SEED,
    "dataset_reader": BASE_READER(ADD_VAE, ADD_ELMO, THROTTLE, USE_SPACY_TOKENIZER),
    "validation_dataset_reader": BASE_READER(ADD_VAE, ADD_ELMO, null, USE_SPACY_TOKENIZER),
    "datasets_for_vocab_creation": ["train"],
    "train_data_path": TRAIN_PATH,
    "validation_data_path": DEV_PATH,
    "vocabulary":{
        "type": "vocabulary_with_vae",
        "vae_vocab_file": "/Users/suching/Github/vae/model_logs/saved_model/vae.txt",
    },
    "model": {
        "type": "logistic_regression",
        "text_field_embedder": EMBEDDER(ADD_VAE, ADD_ELMO)
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
        "cuda_device": if NUM_GPUS == 0 then -1 else if NUM_GPUS > 1 then std.range(0, NUM_GPUS - 1) else 0,
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 0
        }
    }
}

