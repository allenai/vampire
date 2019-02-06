local NUM_GPUS = 0;
// throttle training data
local THROTTLE = null;
local VOCAB_SIZE = 30000;
local LATENT_DIM = 128;
local HIDDEN_DIM = 512;
local ADD_ELMO = false;
local TRAIN_PATH = "s3://suching-dev/imdb/train.jsonl";
local DEV_PATH = "s3://suching-dev/imdb/dev.jsonl";
local REFERENCE_COUNTS = "s3://suching-dev/valid_npmi_reference/train.npz";
local REFERENCE_VOCAB = "s3://suching-dev/valid_npmi_reference/train.vocab.json";
local STOPWORDS_PATH = "s3://suching-dev/stopwords/snowball_stopwords.txt";
local TRACK_TOPICS = true;
local TRACK_NPMI = true;
local KL_WEIGHT_ANNEALING = "sigmoid";
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
                "stopword_file": STOPWORDS_PATH
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
    "random_seed": std.extVar("SEED"),
    "numpy_seed": std.extVar("SEED"),
    "pytorch_seed": std.extVar("SEED"),
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
  "datasets_for_vocab_creation": ["train"],
    "model": {
      "type": "nvdm",
      "apply_batchnorm": true,
      "update_background_freq": false,
      "kl_weight_annealing": KL_WEIGHT_ANNEALING,
      "reference_counts": REFERENCE_COUNTS,
      "reference_vocabulary": REFERENCE_VOCAB,
      "bow_embedder": {
          "type": "bag_of_word_counts",
          "vocab_namespace": "vae",
          "ignore_oov": true
      },
      "vae": {
        "encoder": {
          "input_dim": VOCAB_SIZE + 2,
          "num_layers": 2,
          "hidden_dims": [HIDDEN_DIM, HIDDEN_DIM],
          "activations": ["softplus", "softplus"]
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
      "batch_size": 128,
      "track_epoch": true
    },
    "trainer": {
      "validation_metric": VALIDATION_METRIC,
      "num_epochs": 200,
      "patience": 10,
      "cuda_device": if NUM_GPUS == 0 then -1 else if NUM_GPUS > 1 then std.range(0, NUM_GPUS - 1) else 0,
      "optimizer": {
        "type": "adam",
        "lr": 0.001
      }
    }
  }
