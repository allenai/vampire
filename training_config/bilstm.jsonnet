local NUM_GPUS = 0;
// throttle training data
local THROTTLE = 100;
// add vae embeddings
local ADD_VAE = false;
local ADD_ELMO = false;
local NUM_LABELS = 2;
local TRAIN_PATH = "s3://suching-dev/imdb/train.jsonl";
local DEV_PATH = "s3://suching-dev/imdb/dev.jsonl";

// set to false during debugging
local USE_SPACY_TOKENIZER = true;


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
                "expand_dim": true,
                "model_archive": "s3://suching-dev/model.tar.gz",
                "background_frequency": "s3://suching-dev/vae.bgfreq.json",
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
        },
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "namespace": "tokens",
                "lowercase_tokens": true,
            } 
        } + if add_vae then VAE_FIELDS['vae_indexer'] else {}
            + if add_elmo then ELMO_FIELDS['elmo_indexer'] else {},
        "sequence_length": 400,
        "sample": throttle,
};

local EMBEDDER(add_vae, add_elmo) = {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 300,
                    "trainable": true
                }
            } + if add_vae then VAE_FIELDS['vae_embedder'] else {}
                + if add_elmo then ELMO_FIELDS['elmo_embedder'] else {}
};

{
    "random_seed": std.extVar("SEED"),
    "numpy_seed": std.extVar("SEED"),
    "pytorch_seed": std.extVar("SEED"),
    "dataset_reader": BASE_READER(ADD_VAE, ADD_ELMO, THROTTLE, USE_SPACY_TOKENIZER),
    "validation_dataset_reader": BASE_READER(ADD_VAE, ADD_ELMO, null, USE_SPACY_TOKENIZER),
  "datasets_for_vocab_creation": ["train"],
  "train_data_path": TRAIN_PATH,
  "validation_data_path": DEV_PATH,
    "model": {
        "type": "seq2seq_classifier",
        "text_field_embedder": EMBEDDER(ADD_VAE, ADD_ELMO),
        "encoder": {
           "type": "lstm",
           "num_layers": 2,
           "bidirectional": true,
	       "input_size": 300,
           "hidden_size": 128, 
        },
        "aggregations": ["maxpool", "final_state"],
        "classification_layer": {
            "input_dim": 512,
            "num_layers": 1,
            "hidden_dims": NUM_LABELS,
            "activations": "linear"
        },
        "initializer": [
            [".*linear_layers.*weight", {"type": "xavier_uniform"}],
            [".*linear_layers.*bias", {"type": "zero"}],
            [".*weight_ih.*", {"type": "xavier_uniform"}],
            [".*weight_hh.*", {"type": "orthogonal"}],
            [".*bias_ih.*", {"type": "zero"}],
            [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
        ]
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 32
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.0001
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 2,
        "num_epochs": 75,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": -1,
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 0
        }
    }
}

