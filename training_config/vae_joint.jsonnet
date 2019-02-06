local NUM_GPUS = 1;
// throttle training data
local THROTTLE = null;
local SEED = 50;
local VAE_VOCAB_SIZE = 30000;
local NUM_LABELS = 2;
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
local KL_WEIGHT_ANNEALING = "linear";
local VALIDATION_METRIC = "+accuracy";
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
  "unrestricted_tokenizer": {
        "word_splitter": if use_spacy_tokenizer then "spacy" else "just_spaces",
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
  "sample": throttle,
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
      "vae": VAE_VOCAB_SIZE,
    }
  },
  "datasets_for_vocab_creation": ["train"],
  "model": {
    "type": "joint_m2_classifier",
    "alpha": 50,
    "apply_batchnorm": true,
    "update_background_freq": false,
    "kl_weight_annealing": KL_WEIGHT_ANNEALING,
    "reference_counts": REFERENCE_COUNTS,
    "reference_vocabulary": REFERENCE_VOCAB,
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
        "input_dim": VAE_VOCAB_SIZE + 4,
        "num_layers": 1,
        "hidden_dims": [HIDDEN_DIM],
        "activations": ["softplus"]
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
        "hidden_dims": [VAE_VOCAB_SIZE + 2],
        "activations": ["tanh"]
      },
      "apply_batchnorm": false,
      // "z_dropout": 0.2
    },
    "classification_layer": {
      "input_dim": 300,
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
      "lr": 0.0005,
    }
  }
}  
  