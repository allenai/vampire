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

local BASE_READER(add_vae) = {
        "lazy": false,
        "type": "semisupervised_text_classification_json",
        "tokenizer": {
            "word_splitter": "spacy",
        },
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "namespace": "tokens",
                "lowercase_tokens": true,
                "end_tokens": ["@@PADDING@@", "@@PADDING@@"],
                "start_tokens": ["@@PADDING@@", "@@PADDING@@"]
            } 
        } + if add_vae then VAE_FIELDS['vae_indexer'] else {},
        "ignore_labels": false,
        "shift_target": false,
        "sequence_length": 400
};

local EMBEDDER(add_vae) = {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 300,
                    "trainable": true
                }
            } + if add_vae then VAE_FIELDS['vae_embedder'] else {}
};

{
    "dataset_reader": BASE_READER(ADD_VAE, throttle),
    "validation_dataset_reader": BASE_READER(ADD_VAE, null),
  "datasets_for_vocab_creation": ["train"],
  "train_data_path": "/home/ubuntu/data/ag-news/100/train.jsonl",
  "validation_data_path": "/home/ubuntu/data/ag-news/full/dev.jsonl",
    "model": {
        "type": "seq2vec_classifier",
        "text_field_embedder": EMBEDDER(ADD_VAE),
        "encoder": {
           "type": "cnn",
           "num_filters": 100,
           "embedding_dim": 300,
           "output_dim": 512, 
        },
        "output_feedforward": {
            "input_dim": 512,
            "num_layers": 1,
            "hidden_dims": 128,
            "activations": "relu",
            "dropout": 0.5
        },
        "classification_layer": {
            "input_dim": 128,
            "num_layers": 1,
            "hidden_dims": 32,
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
            "lr": 0.0004
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

