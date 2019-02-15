local CUDA_DEVICE =
  if std.parseInt(std.extVar("NUM_GPU")) == 0 then
    -1
  else if std.parseInt(std.extVar("NUM_GPU")) > 1 then
    std.range(0, std.parseInt(std.extVar("NUM_GPU")) - 1)
  else if std.parseInt(std.extVar("NUM_GPU")) == 1 then
    0;

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

local BASE_READER(ADD_ELMO, THROTTLE, UNLABELED_DATA_PATH, USE_SPACY_TOKENIZER) = {
  "lazy": false,
  "type": "semisupervised_text_classification_json",
  "tokenizer": {
    "word_splitter": if USE_SPACY_TOKENIZER == 1 then "spacy" else "just_spaces",
    "word_filter": {
      "type": "regex_and_stopwords",
      "patterns": [
        "\\w{1,3}\\b", // tokens of length <= 3
        "\\w*\\d+\\w*", // words that contain digits,
         "\\w*[^\\P{P}]+\\w*" // punctuation
      ],
      "tokens_to_add": [">", "<"],
      "stopword_file": std.extVar("STOPWORDS_PATH")
    }
  },
  "token_indexers": {
    "tokens": {
      "type": "single_id",
      "namespace": "vae",
      "lowercase_tokens": true
    }
  } + if ADD_ELMO == 1 then ELMO_FIELDS['elmo_indexer'] else {},
  "sequence_length": 400,
  "ignore_labels": true,
  "sample": THROTTLE,
  "unlabeled_data_path": UNLABELED_DATA_PATH,

};


{
   "numpy_seed": std.extVar("SEED"),
   "pytorch_seed": std.extVar("SEED"),
   "random_seed": std.extVar("SEED"),
   "dataset_reader": BASE_READER(std.parseInt(std.extVar("ADD_ELMO")), std.extVar("THROTTLE"), std.extVar("UNLABELED_DATA_PATH"), std.parseInt(std.extVar("USE_SPACY_TOKENIZER"))),
    "validation_dataset_reader": BASE_READER(std.parseInt(std.extVar("ADD_ELMO")), null, null, std.parseInt(std.extVar("USE_SPACY_TOKENIZER"))),
   "datasets_for_vocab_creation": [
      "train"
   ],
   "train_data_path": std.extVar("TRAIN_PATH"),
   "validation_data_path": std.extVar("DEV_PATH"),
   "vocabulary": {
      "max_vocab_size": {
         "vae": std.parseInt(std.extVar("VOCAB_SIZE"))
      },
      "type": "bg_dumper"
   },
   "model": {
      "apply_batchnorm": true,
      "bow_embedder": {
         "type": "bag_of_word_counts",
         "vocab_namespace": "vae",
         "ignore_oov": true
      },
      "kl_weight_annealing": std.extVar("KL_ANNEALING"),
      "reference_counts": std.extVar("REFERENCE_COUNTS"),
      "reference_vocabulary": std.extVar("REFERENCE_VOCAB"),
      "type": "nvdm",
      "update_background_freq": false,
      "vae": {
         "z_dropout": std.parseInt(std.extVar("Z_DROPOUT")) / 10.0,
         "apply_batchnorm": false,
         "encoder": {
            "activations": [
               "softplus" for x in std.range(0, std.parseInt(std.extVar("NUM_ENCODER_LAYERS")) - 1)
            ],
            "hidden_dims": [
               std.parseInt(std.extVar("VAE_HIDDEN_DIM")) for x in  std.range(0, std.parseInt(std.extVar("NUM_ENCODER_LAYERS")) - 1)
            ],
            "input_dim": std.parseInt(std.extVar("VOCAB_SIZE")) + 2,
            "num_layers": std.parseInt(std.extVar("NUM_ENCODER_LAYERS"))
         },
         "mean_projection": {
            "activations": [
               "linear"
            ],
            "hidden_dims": [
               std.parseInt(std.extVar("VAE_HIDDEN_DIM"))
            ],
            "input_dim": std.extVar("VAE_HIDDEN_DIM"),
            "num_layers": 1
         },
        "log_variance_projection": {
            "activations": [
               "linear"
            ],
            "hidden_dims": [
               std.parseInt(std.extVar("VAE_HIDDEN_DIM"))
            ],
            "input_dim": std.parseInt(std.extVar("VAE_HIDDEN_DIM")),
            "num_layers": 1
         },
         "decoder": {
            "activations": [
               "tanh"
            ],
            "hidden_dims": [
               std.parseInt(std.extVar("VOCAB_SIZE")) + 2
            ],
            "input_dim": std.parseInt(std.extVar("VAE_HIDDEN_DIM")),
            "num_layers": 1
         },
         "type": "logistic_normal"
      }
   },
    "iterator": {
      "batch_size": 128,
      "track_epoch": true,
      "type": "basic"
   },
   "trainer": {
      "cuda_device": CUDA_DEVICE,
      "num_epochs": 200,
      "optimizer": {
         "lr": std.parseInt(std.extVar("LEARNING_RATE")) / 10000.0,
         "type": "adam"
      },
      "patience": 75,
      "validation_metric": std.extVar("VALIDATION_METRIC")
   }
}
