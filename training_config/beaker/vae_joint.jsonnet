// TODO: Abstract all std calls into jsonnet local variables may help readability.

local CUDA_DEVICE =
  if std.parseInt(std.extVar("NUM_GPU")) == 0 then
    -1
  else if std.parseInt(std.extVar("NUM_GPU")) > 1 then
    std.range(0, std.extVar("NUM_GPU") - 1)
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

local BASE_JOINT_READER(ADD_ELMO, THROTTLE, UNLABELED_DATA_PATH, USE_SPACY_TOKENIZER, UNLABELED_DATA_FACTOR) = {
  "lazy": true,
  "type": "joint_semisupervised_text_classification_json",
  "tokenizer": {
    "word_splitter": if USE_SPACY_TOKENIZER == 1 then "spacy" else "just_spaces",
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
        "word_splitter": if USE_SPACY_TOKENIZER == 1 then "spacy" else "just_spaces",
    },
  "token_indexers": {
    "classifier_tokens": {
      "type": "single_id",
      "namespace": "classifier",
      "lowercase_tokens": true
    },
    "tokens": {
      "type": "single_id",
      "namespace": "vae",
      "lowercase_tokens": true
    }
  } + if ADD_ELMO == 1 then ELMO_FIELDS['elmo_indexer'] else {},
  "sequence_length": 400,
  "sample": THROTTLE,
  "unlabeled_data_path": UNLABELED_DATA_PATH,
  "unlabeled_data_factor": UNLABELED_DATA_FACTOR
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
         "\\w*[^\\P{P}\\-]+\\w*" // punctuation
      ],
      "stopword_file": std.extVar("STOPWORDS_PATH")
    }
  },
  "unrestricted_tokenizer": {
        "word_splitter": if USE_SPACY_TOKENIZER == 1 then "spacy" else "just_spaces",
    },
  "token_indexers": {
    "classifier_tokens": {
      "type": "single_id",
      "namespace": "classifier",
      "lowercase_tokens": true
    },
    "tokens": {
      "type": "single_id",
      "namespace": "vae",
      "lowercase_tokens": true
    }
  } + if ADD_ELMO == 1 then ELMO_FIELDS['elmo_indexer'] else {},
  "sequence_length": 400,
  "sample": THROTTLE,
  "unlabeled_data_path": UNLABELED_DATA_PATH
};


local BOE_CLF(EMBEDDING_DIM, ADD_ELMO) = {
         "encoder": {
            "type": "seq2vec",
             "architecture": {
                "embedding_dim": EMBEDDING_DIM,
                "type": "boe"
             }
         },
         "input_embedder": {
            "token_embedders": {
               "tokens": {
                  "embedding_dim": EMBEDDING_DIM,
                  "trainable": true,
                  "type": "embedding",
                  "vocab_namespace": "classifier"
               }
            } + if ADD_ELMO == 1 then ELMO_FIELDS['elmo_embedder'] else {}
         },
         
      
};

local CNN_CLF(EMBEDDING_DIM, NUM_FILTERS, CLF_HIDDEN_DIM, ADD_ELMO) = {
         "encoder": {
             "type": "seq2vec",
             "architecture": {
                 "type": "cnn",
                 "num_filters": NUM_FILTERS,
                 "embedding_dim": EMBEDDING_DIM,
                 "output_dim": CLF_HIDDEN_DIM, 
             }
         },
         "input_embedder": {
            "token_embedders": {
               "tokens": {
                  "embedding_dim": EMBEDDING_DIM,
                  "trainable": true,
                  "type": "embedding",
                  "vocab_namespace": "classifier"
               }
            }
         } + if ADD_ELMO == 1 then ELMO_FIELDS['elmo_embedder'] else {},

      
};

local LSTM_CLF(EMBEDDING_DIM, NUM_CLF_ENCODER_LAYERS, CLF_HIDDEN_DIM, AGGREGATIONS, ADD_ELMO) = {
        "input_embedder": {
            "token_embedders": {
               "tokens": {
                  "embedding_dim": EMBEDDING_DIM,
                  "trainable": true,
                  "type": "embedding",
                  "vocab_namespace": "classifier"
               }
            } + if ADD_ELMO == 1 then ELMO_FIELDS['elmo_embedder'] else {}
         },
        "encoder": {
          "type" : "seq2seq",
          "architecture": {
            "type": "lstm",
            "num_layers": NUM_CLF_ENCODER_LAYERS,
            "bidirectional": false,
            "input_size": EMBEDDING_DIM,
            "hidden_size": CLF_HIDDEN_DIM
          },
         "aggregations": AGGREGATIONS,
        },

};

local LR_CLF() = {
        "input_embedder": {
            "token_embedders": {
               "tokens": {
                  "type": "bag_of_word_counts",
                  "ignore_oov": true,
                  "vocab_namespace": "classifier"
               }
            }
         }
};

local CLASSIFIER = 
    if std.extVar("CLASSIFIER") == "lstm" then
        LSTM_CLF(std.parseInt(std.extVar("EMBEDDING_DIM")),
                 std.parseInt(std.extVar("NUM_CLF_ENCODER_LAYERS")),
                 std.parseInt(std.extVar("CLF_HIDDEN_DIM")),
                 std.extVar("AGGREGATIONS"),
                 std.parseInt(std.extVar("ADD_ELMO")))
    else if std.extVar("CLASSIFIER") == "cnn" then
        CNN_CLF(std.parseInt(std.extVar("EMBEDDING_DIM")),
                std.parseInt(std.extVar("NUM_FILTERS")),
                std.parseInt(std.extVar("CLF_HIDDEN_DIM")),
                std.parseInt(std.extVar("ADD_ELMO")))
    else if std.extVar("CLASSIFIER") == "boe" then
        BOE_CLF(std.parseInt(std.extVar("EMBEDDING_DIM")),
                std.parseInt(std.extVar("ADD_ELMO")))
    else if std.extVar("CLASSIFIER") == 'lr' then
        LR_CLF();




{
   "numpy_seed": std.extVar("SEED"),
   "pytorch_seed": std.extVar("SEED"),
   "random_seed": std.extVar("SEED"),
   "dataset_reader": BASE_JOINT_READER(std.parseInt(std.extVar("ADD_ELMO")), std.extVar("THROTTLE"), std.extVar("UNLABELED_DATA_PATH"), std.parseInt(std.extVar("USE_SPACY_TOKENIZER")), std.extVar("UNLABELED_DATA_FACTOR")),
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
      "alpha": std.parseInt(std.extVar("ALPHA")),
      "apply_batchnorm": true,
      "bow_embedder": {
         "type": "bag_of_word_counts",
         "vocab_namespace": "vae",
         "ignore_oov": true
      },
      "kl_weight_annealing": std.extVar("KL_ANNEALING"),
      "reference_counts": std.extVar("REFERENCE_COUNTS"),
      "reference_vocabulary": std.extVar("REFERENCE_VOCAB"),
      "type": "joint_m2_classifier",
      "update_background_freq": false,
      "classifier": CLASSIFIER,
      "vae": {
         "apply_batchnorm": false,
         "encoder": {
            "activations": [
               "softplus" for x in std.range(0, std.parseInt(std.extVar("NUM_VAE_ENCODER_LAYERS")) - 1)
            ],
            "hidden_dims": [
               std.parseInt(std.extVar("VAE_HIDDEN_DIM")) for x in  std.range(0, std.parseInt(std.extVar("NUM_VAE_ENCODER_LAYERS")) - 1)
            ],
            "input_dim": std.parseInt(std.extVar("VOCAB_SIZE")) + 4,
            "num_layers": std.parseInt(std.extVar("NUM_VAE_ENCODER_LAYERS"))
         },
         "mean_projection": {
            "activations": [
               "linear"
            ],
            "hidden_dims": [
               std.parseInt(std.extVar("VAE_LATENT_DIM"))
            ],
            "input_dim": std.extVar("VAE_HIDDEN_DIM"),
            "num_layers": 1
         },
        "log_variance_projection": {
            "activations": [
               "linear"
            ],
            "hidden_dims": [
               std.parseInt(std.extVar("VAE_LATENT_DIM"))
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
            "input_dim": std.parseInt(std.extVar("VAE_LATENT_DIM")),
            "num_layers": 1
         },
         "type": "logistic_normal"
      }
   },
    "iterator": {
      "batch_size": std.parseInt(std.extVar("BATCH_SIZE")),
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
      "patience": 20,
      "validation_metric": "+accuracy"
   }
}
