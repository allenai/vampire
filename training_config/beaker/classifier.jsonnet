local CUDA_DEVICE =
  if std.parseInt(std.extVar("NUM_GPU")) == 0 then
    -1
  else if std.parseInt(std.extVar("NUM_GPU")) > 1 then
    std.range(0, std.extVar("NUM_GPU") - 1)
  else if std.parseInt(std.extVar("NUM_GPU")) == 1 then
    0;


local BERT_FIELDS = {
  "bert_indexer": {
       "bert": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased"
    }
  },
  "bert_embedder": {
    "bert": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased",
        "requires_grad": std.parseInt(std.extVar('BERT_FINETUNE')) == 1,
        "top_layer_only": false
        }
  },
  "extra_embedder_fields": {
    "allow_unmatched_keys": true,
    "embedder_to_indexer_map": {
        "bert": ["bert", "bert-offsets"],
        "tokens": ["tokens"]
    }
  },
};

// local TRANSFORMER_ELMO_FIELDS = {
//   "elmo_indexer": {
//     "elmo": {
//       "type": "elmo_characters",
//     }
//   },
//   "elmo_embedder": {
//     "elmo":{
//             "type": "elmo_token_embedder",
//             "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
//             "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
//             "do_layer_norm": false,
//             "dropout": 0.0,
//             "scalar_mix_parameters": [0.0, 0.0, 20.0],
//             "requires_grad": std.parseInt(std.extVar("ELMO_FINETUNE")) == 1,
//     },
//   }
// };

local ELMO_OPTIMIZER = {
      "optimizer": {
                "type": "adam",
                "lr": 0.004,
                "parameter_groups": [
                      [["_input_embedder.token_embedder_elmo._lm._text_field_embedder.token_embedder_token_characters.*"], {}],
                      [["_input_embedder.token_embedder_elmo._lm._contextualizer._backward_transformer.*"], {}],
                      [["_input_embedder.token_embedder_elmo._lm._contextualizer._forward_transformer.*"], {}],
                      // [["_input_embedder.token_embedder_elmo._elmo._elmo_lstm._elmo_lstm.forward_layer_0.*", "_input_embedder.token_embedder_elmo._elmo._elmo_lstm._elmo_lstm.backward_layer_0.*"], {}],
                      // [["_input_embedder.token_embedder_elmo._elmo._elmo_lstm._elmo_lstm.forward_layer_1.*", "_input_embedder.token_embedder_elmo._elmo._elmo_lstm._elmo_lstm.backward_layer_1.*"], {}],
                      [["^_classification_layer.weight", "^_classification_layer.bias", ".*scalar_mix.*"], {}]
                ],
    },
    "learning_rate_scheduler": {
        "type": "slanted_triangular",
        "gradual_unfreezing": true,
        "discriminative_fine_tuning": true,
        "num_epochs": 50,
        "ratio": 32,
        "decay_factor": 0.4,
        // 98794 training instances for use-trees and sst-2
        "num_steps_per_epoch": 313,
    }
};

local TRANSFORMER_ELMO_FIELDS = {
  "elmo_indexer": {
    "elmo": {
      "type": "elmo_characters",
    }
  },
  "elmo_embedder": {
    "elmo": {
      "type": "bidirectional_lm_token_embedder",
      "archive_file": std.extVar('ELMO_ARCHIVE_PATH'),
      "dropout": std.parseInt(std.extVar("ELMO_DROPOUT")) / 10.0,
      "bos_eos_tokens": ["<S>", "</S>"],
      "remove_bos_eos": true,
      "requires_grad": std.parseInt(std.extVar("ELMO_FINETUNE")) == 1
    }
  }
};


local BASIC_FIELDS(EMBEDDING_DIM) = {
  "basic_indexer": {
    "tokens": {
      "type": "single_id",
      "lowercase_tokens": true,
      "namespace": "classifier"
    }
  },
  "basic_embedder": {
    "tokens": {
         "embedding_dim": EMBEDDING_DIM,
        "trainable": true,
        "type": "embedding",
        "vocab_namespace": "classifier"
    }
  },
};

local GLOVE_FIELDS = {
  "glove_indexer": {
    "tokens": {
      "type": "single_id",
      "lowercase_tokens": true,
      "namespace": "classifier"
    }
  },
  "glove_embedder": {
    "tokens": {
        "embedding_dim": 50,
        "trainable": true,
        "pretrained_file": std.extVar("GLOVE_PATH"),
        "vocab_namespace": "classifier"
    }
  },
};


local VAE_FIELDS(EXPAND_DIM) = {
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
                "scalar_mix": [1, -20, 1],
                "expand_dim": EXPAND_DIM,
                "requires_grad": std.parseInt(std.extVar("VAE_FINETUNE")) == 1,
                "model_archive": std.extVar("VAE_MODEL_ARCHIVE"),
                "background_frequency": std.extVar("VAE_BG_FREQ"),
                "dropout": std.parseInt(std.extVar("VAE_DROPOUT")) / 10.0
        }
    }
};

local VOCABULARY_WITH_VAE = {
  "vocabulary":{
              "type": "vocabulary_with_vae",
              "vae_vocab_file": std.extVar("VAE_VOCAB"),
          }
};

local VAE_INDEXER = if std.parseInt(std.extVar("ADD_VAE")) == 1 then VAE_FIELDS(true)['vae_indexer'] else {};
local ELMO_INDEXER = if std.parseInt(std.extVar("ADD_ELMO")) == 1 then TRANSFORMER_ELMO_FIELDS['elmo_indexer'] else {};
local BERT_INDEXER = if std.parseInt(std.extVar("ADD_BERT")) == 1 then BERT_FIELDS['bert_indexer'] else {};
local BASIC_INDEXER = if std.parseInt(std.extVar("ADD_BASIC")) == 1 then BASIC_FIELDS(std.parseInt(std.extVar("EMBEDDING_DIM")))['basic_indexer'] else {};
local GLOVE_INDEXER = if std.parseInt(std.extVar("ADD_GLOVE")) == 1 then GLOVE_FIELDS['glove_indexer'] else {};


local BASE_READER(VAE_INDEXER,ELMO_INDEXER, BERT_INDEXER, BASIC_INDEXER, GLOVE_INDEXER, THROTTLE, USE_SPACY_TOKENIZER) = {
  "lazy": false,
  "type": "semisupervised_text_classification_json",
  "tokenizer": {
    "word_splitter": if USE_SPACY_TOKENIZER == 1 then "spacy" else "just_spaces",
  },
  "token_indexers": {} + VAE_INDEXER + ELMO_INDEXER + BERT_INDEXER + BASIC_INDEXER + GLOVE_INDEXER,
  "sequence_length": 400,
  "sample": THROTTLE,
};




local VAE_EMBEDDINGS = if std.parseInt(std.extVar("ADD_VAE")) == 1 then VAE_FIELDS(true)['vae_embedder'] else {};
local ELMO_EMBEDDINGS = if std.parseInt(std.extVar("ADD_ELMO")) == 1 then TRANSFORMER_ELMO_FIELDS['elmo_embedder'] else {};
local BERT_EMBEDDINGS = if std.parseInt(std.extVar("ADD_BERT")) == 1 then BERT_FIELDS['bert_embedder'] else {};
local BASIC_EMBEDDINGS = if std.parseInt(std.extVar("ADD_BASIC")) == 1 then BASIC_FIELDS(std.parseInt(std.extVar("EMBEDDING_DIM")))['basic_embedder'] else {};
local GLOVE_EMBEDDINGS = if std.parseInt(std.extVar("ADD_GLOVE")) == 1 then GLOVE_FIELDS['glove_embedder'] else {};


local BOE_CLF(EMBEDDING_DIM, ENCODER_INPUT_DIM, BASIC_EMBEDDINGS, BERT_EMBEDDINGS, ELMO_EMBEDDINGS , VAE_EMBEDDINGS, GLOVE_EMBEDDINGS, ADD_BERT) = {
         "input_embedder": {
            "token_embedders": {} + BASIC_EMBEDDINGS + BERT_EMBEDDINGS + ELMO_EMBEDDINGS + VAE_EMBEDDINGS + GLOVE_EMBEDDINGS
         } + if ADD_BERT == 1 then BERT_FIELDS['extra_embedder_fields'] else {},
         
         "encoder": {
            "type": "seq2vec",
             "architecture": {
                "embedding_dim": ENCODER_INPUT_DIM,
                "type": "boe",
                "averaged": true
             }
         },
         "dropout": std.parseInt(std.extVar("DROPOUT")) / 10
};

local MAXPOOL_CLF(EMBEDDING_DIM, ENCODER_INPUT_DIM, BASIC_EMBEDDINGS, BERT_EMBEDDINGS, ELMO_EMBEDDINGS , VAE_EMBEDDINGS, GLOVE_EMBEDDINGS, ADD_BERT) = {
         "input_embedder": {
            "token_embedders": {} + BASIC_EMBEDDINGS + BERT_EMBEDDINGS + ELMO_EMBEDDINGS + VAE_EMBEDDINGS + GLOVE_EMBEDDINGS
         } + if ADD_BERT == 1 then BERT_FIELDS['extra_embedder_fields'] else {},
         
         "encoder": {
            "type": "seq2vec",
             "architecture": {
                "embedding_dim": ENCODER_INPUT_DIM,
                "type": "maxpool",
             }
         },
         "dropout": std.parseInt(std.extVar("DROPOUT")) / 10
};



local CNN_CLF(EMBEDDING_DIM, ENCODER_INPUT_DIM, NUM_FILTERS, CLF_HIDDEN_DIM, BASIC_EMBEDDINGS, BERT_EMBEDDINGS, ELMO_EMBEDDINGS , VAE_EMBEDDINGS, GLOVE_EMBEDDINGS, ADD_BERT) = {
         
         "input_embedder": {
            "token_embedders": {} + BASIC_EMBEDDINGS + BERT_EMBEDDINGS + ELMO_EMBEDDINGS + VAE_EMBEDDINGS + GLOVE_EMBEDDINGS
         } + if ADD_BERT == 1 then BERT_FIELDS['extra_embedder_fields'] else {},
         "encoder": {
             "type": "seq2vec",
             "architecture": {
                 "type": "cnn",
                 "ngram_filter_sizes": std.range(1, std.parseInt(std.extVar("MAX_FILTER_SIZE"))),
                 "num_filters": NUM_FILTERS,
                 "embedding_dim": ENCODER_INPUT_DIM,
                 "output_dim": CLF_HIDDEN_DIM, 
             },
         },
         "dropout": std.parseInt(std.extVar("DROPOUT")) / 10
};

local LSTM_CLF(EMBEDDING_DIM, ENCODER_INPUT_DIM, NUM_ENCODER_LAYERS, CLF_HIDDEN_DIM, AGGREGATIONS, BASIC_EMBEDDINGS, BERT_EMBEDDINGS, ELMO_EMBEDDINGS , VAE_EMBEDDINGS, GLOVE_EMBEDDINGS, ADD_BERT) = {
        "input_embedder": {
            "token_embedders": {} + BASIC_EMBEDDINGS + BERT_EMBEDDINGS + ELMO_EMBEDDINGS + VAE_EMBEDDINGS + GLOVE_EMBEDDINGS
         } + if ADD_BERT == 1 then BERT_FIELDS['extra_embedder_fields'] else {},
        "encoder": {
          "type" : "seq2seq",
          "architecture": {
            "type": "lstm",
            "num_layers": NUM_ENCODER_LAYERS,
            "bidirectional": true,
            "input_size": ENCODER_INPUT_DIM,
            "hidden_size": CLF_HIDDEN_DIM
          },
         "aggregations": AGGREGATIONS,
        },
        "dropout": std.parseInt(std.extVar("DROPOUT")) / 10
};

local LR_CLF(ADD_VAE) = {
        "input_embedder": {
            "token_embedders": {
               "tokens": {
                  "type": "bag_of_word_counts",
                  "ignore_oov": "true",
                  "vocab_namespace": "classifier"
               }
            } + if ADD_VAE == 1 then VAE_FIELDS(false)['vae_embedder'] else {}
         },
         "dropout": std.parseInt(std.extVar("DROPOUT")) / 10
};

local ENCODER_INPUT_DIM = std.parseInt(std.extVar("ENCODER_INPUT_DIM"));

local CLASSIFIER = 
    if std.extVar("CLASSIFIER") == "lstm" then
        LSTM_CLF(std.parseInt(std.extVar("EMBEDDING_DIM")),
                 ENCODER_INPUT_DIM,
                 std.parseInt(std.extVar("NUM_ENCODER_LAYERS")),
                 std.parseInt(std.extVar("CLF_HIDDEN_DIM")),
                 std.extVar("AGGREGATIONS"),
                 BASIC_EMBEDDINGS,
                 BERT_EMBEDDINGS,
                 ELMO_EMBEDDINGS,
                 VAE_EMBEDDINGS,
                 GLOVE_EMBEDDINGS,
                 std.parseInt(std.extVar("ADD_BERT")))
    else if std.extVar("CLASSIFIER") == "cnn" then
        CNN_CLF(std.parseInt(std.extVar("EMBEDDING_DIM")),
                ENCODER_INPUT_DIM,
                std.parseInt(std.extVar("NUM_FILTERS")),
                std.parseInt(std.extVar("CLF_HIDDEN_DIM")),
                BASIC_EMBEDDINGS,
                BERT_EMBEDDINGS,
                ELMO_EMBEDDINGS,
                VAE_EMBEDDINGS,
                GLOVE_EMBEDDINGS,
                std.parseInt(std.extVar("ADD_BERT")))
    else if std.extVar("CLASSIFIER") == "boe" then
        BOE_CLF(std.parseInt(std.extVar("EMBEDDING_DIM")),
                ENCODER_INPUT_DIM,
                BASIC_EMBEDDINGS,
                BERT_EMBEDDINGS,
                ELMO_EMBEDDINGS,
                VAE_EMBEDDINGS,
                GLOVE_EMBEDDINGS,
                std.parseInt(std.extVar("ADD_BERT")))
    else if std.extVar("CLASSIFIER") == "maxpool" then
        MAXPOOL_CLF(std.parseInt(std.extVar("EMBEDDING_DIM")),
                ENCODER_INPUT_DIM,
                BASIC_EMBEDDINGS,
                BERT_EMBEDDINGS,
                ELMO_EMBEDDINGS,
                VAE_EMBEDDINGS,
                GLOVE_EMBEDDINGS,
                std.parseInt(std.extVar("ADD_BERT")))
    else if std.extVar("CLASSIFIER") == 'lr' then
        LR_CLF(std.parseInt(std.extVar("ADD_VAE")));

{
   "numpy_seed": std.extVar("SEED"),
   "pytorch_seed": std.extVar("SEED"),
   "random_seed": std.extVar("SEED"),
   "evaluate_on_test": std.parseInt(std.extVar("EVALUATE_ON_TEST")) == 1,
   "dataset_reader": BASE_READER(VAE_INDEXER,ELMO_INDEXER, BERT_INDEXER, BASIC_INDEXER, GLOVE_INDEXER, std.extVar("THROTTLE"), std.parseInt(std.extVar("USE_SPACY_TOKENIZER"))),
    "validation_dataset_reader": BASE_READER(VAE_INDEXER,ELMO_INDEXER, BERT_INDEXER, BASIC_INDEXER, GLOVE_INDEXER, null, std.parseInt(std.extVar("USE_SPACY_TOKENIZER"))),
   "datasets_for_vocab_creation": ["train"],
   "train_data_path": std.extVar("TRAIN_PATH"),
   "validation_data_path": std.extVar("DEV_PATH"),
   "test_data_path": std.extVar("TEST_PATH"),
   "model": {"type": "classifier"} + CLASSIFIER,
    "iterator": {
      "batch_size": std.parseInt(std.extVar("BATCH_SIZE")),
      "type": "basic"
   },
   "trainer": {
      "cuda_device": CUDA_DEVICE,
      "num_epochs": 50,
      // "optimizer": ELMO_OPTIMIZER['optimizer'],
      "optimizer": {
         "lr": std.parseInt(std.extVar("LEARNING_RATE")) / 10000.0,
         "type": "adam"
      },
      "patience": 5,
      "validation_metric": "+accuracy",
      // "learning_rate_scheduler": ELMO_OPTIMIZER['learning_rate_scheduler'],
   }
} + if std.parseInt(std.extVar("ADD_VAE")) == 1 then VOCABULARY_WITH_VAE else {}
