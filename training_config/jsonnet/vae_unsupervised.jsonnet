local NUM_GPUS = 1;
// throttle training data
local THROTTLE = null;
local SEED = 213;
local VOCAB_SIZE = 20000;
local LATENT_DIM = 100;
local HIDDEN_DIM = 300;
local ADD_ELMO = false;
local TRAIN_PATH = "/home/ubuntu/vae/datasets/imdb/train.jsonl";
local DEV_PATH = "/home/ubuntu/vae/datasets/imdb/dev.jsonl";
local REFERENCE_DIRECTORY = "/home/ubuntu/vae/preprocessed_imdb/";
local TRACK_TOPICS = true;
local TRACK_NPMI = true;
local VALIDATION_METRIC = "+npmi";
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
                            "\\w{1,3}\\b", // tokens of length <= 3
                            "\\w*\\d+\\w*", // words that contain digits,
                             "\\w*[^\\P{P}\\-]+\\w*" // punctuation
                            ],
                "stopword_file": "/home/ubuntu/vae/vae/common/stopwords/snowball_stopwords.txt"
            }
        },
        "token_indexers": {
            "tokens": {
              "type": "single_id",
              "namespace": "vae",
              "lowercase_tokens": true
          }
        } + if add_elmo then ELMO_FIELDS['elmo_indexer'] else {}, 
        "sequence_length": 400,
        "sample": throttle,
};


{
    "random_seed": SEED,
    "numpy_seed": SEED,
    "pytorch_seed": SEED,
    "dataset_reader": BASE_READER(ADD_ELMO, THROTTLE, USE_SPACY_TOKENIZER),
    "validation_dataset_reader": BASE_READER(ADD_ELMO, null, USE_SPACY_TOKENIZER),
  "train_data_path": TRAIN_PATH,
  "validation_data_path": DEV_PATH,
  "vocabulary":{
    "type": "bg_dumper",
    "max_vocab_size": {
      "vae": VOCAB_SIZE,
    }
  },
    "model": {
      "type": "nvdm",
      "update_background_freq": true,
      "ref_directory": REFERENCE_DIRECTORY,
      "bow_embedder": {
          "type": "bag_of_word_counts",
          "vocab_namespace": "vae"
      },
      "vae": {
        "encoder": {
          "input_dim": VOCAB_SIZE + 2,
          "num_layers": 1,
          "hidden_dims": [HIDDEN_DIM],
          "activations": "softplus"
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
          "hidden_dims": [VOCAB_SIZE + 2],
          "activations": ["linear"]
        },
        "apply_batchnorm": false,
        "z_dropout": 0.2
      }
    },
    "iterator": {
      "type": "basic",
      "batch_size": 100,
      "track_epoch": true
    },
    "trainer": {
      "validation_metric": VALIDATION_METRIC,
      "num_epochs": 200,
      "cuda_device": 0,
      "optimizer": {
        "type": "adam",
        "lr": 0.002
      }
    }
  }
    
  
