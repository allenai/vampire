local VAE_VOCAB_SIZE = 2000;
local CLASSIFIER_VOCAB_SIZE = 10000;
local NUM_LABELS = 2;
local HIDDEN_DIM = 300;
local LATENT_DIM = 10;
local THROTTLE = null;
local ADD_ELMO = false;
local TRAIN_PATH = "/home/ubuntu/vae/datasets/imdb/train.jsonl";
local DEV_PATH = "/home/ubuntu/vae/datasets/imdb/dev.jsonl";
// set to false during debugging
local USE_SPACY_TOKENIZER = false;

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

local BASE_READER(add_elmo, throttle, use_spacy_tokenizer) = {
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
                "stopword_file": "/home/ubuntu/vae/vae/common/stopwords/snowball_stopwords.txt"
            }
        },
        "unrestricted_tokenizer": {
        "word_splitter": "spacy"
        },
        "token_indexers": {
            "tokens": {
              "type": "single_id",
              "namespace": "classifier",
              "lowercase_tokens": true
          },
          "filtered_tokens": {
            "type": "single_id",
            "namespace": "vae",
            "lowercase_tokens": true
            }
        } + if add_elmo then ELMO_FIELDS['elmo_indexer'] else {}, 
        "sequence_length": 400,
        "sample": THROTTLE,
};

local EMBEDDER(add_elmo) = {
            "token_embedders": {
                    "tokens": {
                    "type": "embedding",
                    "embedding_dim": 300,
                    "trainable": true,
                    "vocab_namespace": "classifier"
                }
            } + if add_elmo then ELMO_FIELDS['elmo_embedder'] else {}
};


{
    "dataset_reader": BASE_READER(ADD_ELMO, THROTTLE, USE_SPACY_TOKENIZER),
    "validation_dataset_reader": BASE_READER(ADD_ELMO, null, USE_SPACY_TOKENIZER),
  "train_data_path": TRAIN_PATH,
  "validation_data_path": DEV_PATH,
  "vocabulary":{
    "type": "bg_dumper",
    "max_vocab_size": {
      "vae": VAE_VOCAB_SIZE,
      "classifier": CLASSIFIER_VOCAB_SIZE
    }
  },
    "model": {
      "type": "joint_m2_classifier",
      "alpha": 50,
      "update_background_freq": false,
      "input_embedder": EMBEDDER(ADD_ELMO),
      "bow_embedder": {
          "type": "bag_of_word_counts",
          "vocab_namespace": "vae"
      },
      "encoder": {
        "type": "boe",
        "embedding_dim": 300
      },
      "vae": {
        "encoder": {
          "input_dim": CLASSIFIER_VOCAB_SIZE + 2,
          "num_layers": 2,
          "hidden_dims": [1000, HIDDEN_DIM],
          "activations": ["relu", "relu"]
        },
        "mean_projection": {
          "input_dim": HIDDEN_DIM,
          "num_layers": 1,
          "hidden_dims": [LATENT_DIM],
          "activations": ["linear"]
        },
        "log_variance_projection": {
          "input_dim": HIDDEN_DIM,
          "num_layers": 1,
          "hidden_dims": [LATENT_DIM],
          "activations": ["linear"]
        },
        "decoder": {
          "input_dim": LATENT_DIM,
          "num_layers": 1,
          "hidden_dims": [CLASSIFIER_VOCAB_SIZE],
          "activations": ["tanh"]
        },
        "apply_batchnorm": true,
        "z_dropout": 0.2
      },
      "classification_layer": {
        "input_dim": HIDDEN_DIM,
        "num_layers": 1,
        "hidden_dims": [NUM_LABELS],
        "activations": ["linear"]
      }
    },
    "iterator": {
      "type": "basic",
      "batch_size": 100,
      "track_epoch": true
    },
    "trainer": {
      "validation_metric": "+accuracy",
      "num_epochs": 200,
      "patience": 20,
      "cuda_device": -1,
      "optimizer": {
        "type": "adam",
        "lr": 0.001,
        "weight_decay": 0.001
      }
    }
  }
    
  