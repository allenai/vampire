local CUDA_DEVICE =
  if std.parseInt(std.extVar("NUM_GPU")) == 0 then
    -1
  else if std.parseInt(std.extVar("NUM_GPU")) > 1 then
    std.range(0, std.extVar("NUM_GPU") - 1)
  else if std.parseInt(std.extVar("NUM_GPU")) == 1 then
    0;

local ELMO_REQUIRES_GRAD = if std.extVar("ELMO_FINETUNE") == 1 then true else false;


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
        "requires_grad": false,
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


local ELMO_FIELDS = {
  "elmo_indexer": {
    "elmo": {
      "type": "elmo_characters",
    }
  },
  "elmo_embedder": {
    "elmo": {
      "type": "elmo_token_embedder",
      "options_file": std.extVar("ELMO_OPTIONS_FILE"),
      "weight_file": std.extVar("ELMO_WEIGHT_FILE"),
      "do_layer_norm": false,
      "requires_grad": ELMO_REQUIRES_GRAD,
      "dropout": std.parseInt(std.extVar("ELMO_DROPOUT")) / 10.0
    }
  }
};





local REQUIRES_GRAD = 
  if std.extVar("VAE_FINE_TUNE") == 1 then
    true
  else false;

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
                "expand_dim": EXPAND_DIM,
                "requires_grad": REQUIRES_GRAD,
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


local BASE_READER(ADD_ELMO, ADD_VAE, ADD_BERT, THROTTLE, USE_SPACY_TOKENIZER) = {
  "lazy": false,
  "type": "semisupervised_text_classification_json",
  "tokenizer": {
    "word_splitter": if USE_SPACY_TOKENIZER == 1 then "spacy" else "just_spaces",
  },
  "token_indexers": {
    "tokens": {
      "type": "single_id",
      "lowercase_tokens": true,
      "namespace": "classifier"
    }
  } + if ADD_VAE == 1 then VAE_FIELDS(true)['vae_indexer'] else {}
    + if ADD_ELMO == 1 then ELMO_FIELDS['elmo_indexer'] else {}
    + if ADD_BERT == 1 then BERT_FIELDS['bert_indexer'] else {},
  "sequence_length": 400,
  "sample": THROTTLE,
};


local BOE_CLF(EMBEDDING_DIM, ENCODER_INPUT_DIM, ADD_ELMO, ADD_VAE, ADD_BERT) = {
         "encoder": {
            "type": "seq2vec",
             "architecture": {
                "embedding_dim": ENCODER_INPUT_DIM,
                "type": "boe",
                "averaged": true
             }
         },
         "dropout": std.parseInt(std.extVar("DROPOUT")) / 10,
         "input_embedder": {
            "token_embedders": {
               "tokens": {
                  "embedding_dim": EMBEDDING_DIM,
                  "trainable": true,
                  "type": "embedding",
                  "vocab_namespace": "classifier"
               }
            } + if ADD_VAE == 1 then VAE_FIELDS(true)['vae_embedder'] else {}
              + if ADD_ELMO == 1 then ELMO_FIELDS['elmo_embedder'] else {}
              + if ADD_BERT == 1 then BERT_FIELDS['bert_embedder'] else {}

         } + if ADD_BERT == 1 then BERT_FIELDS['extra_embedder_fields'] else {},
         
      
};


local CNN_CLF(EMBEDDING_DIM, ENCODER_INPUT_DIM, NUM_FILTERS,  CLF_HIDDEN_DIM, ADD_ELMO, ADD_VAE, ADD_BERT) = {
         
         "dropout": std.parseInt(std.extVar("DROPOUT")) / 10,
         "input_embedder": {
            "token_embedders": {
               "tokens": {
                  "embedding_dim": EMBEDDING_DIM,
                  "trainable": true,
                  "type": "embedding",
                  "vocab_namespace": "classifier"
               }
            }  + if ADD_VAE == 1 then VAE_FIELDS(true)['vae_embedder'] else {}
               + if ADD_ELMO == 1 then ELMO_FIELDS['elmo_embedder'] else {}
               + if ADD_BERT == 1 then BERT_FIELDS['bert_embedder'] else {}
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

      
};

local LSTM_CLF(EMBEDDING_DIM, ENCODER_INPUT_DIM, NUM_ENCODER_LAYERS, CLF_HIDDEN_DIM, AGGREGATIONS, ADD_ELMO, ADD_VAE, ADD_BERT) = {
        "input_embedder": {
            "token_embedders": {
               "tokens": {
                  "embedding_dim": EMBEDDING_DIM,
                  "trainable": true,
                  "type": "embedding",
                  "vocab_namespace": "classifier"
               }
            } + if ADD_VAE == 1 then VAE_FIELDS(true)['vae_embedder'] else {}
              + if ADD_ELMO == 1 then ELMO_FIELDS['elmo_embedder'] else {}
              + if ADD_BERT == 1 then BERT_FIELDS['bert_embedder'] else {}
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

local ENCODER_INPUT_DIM = std.parseInt(std.extVar("ENCODER_ADDITIONAL_DIM")) + std.parseInt(std.extVar("EMBEDDING_DIM"));

local CLASSIFIER = 
    if std.extVar("CLASSIFIER") == "lstm" then
        LSTM_CLF(std.parseInt(std.extVar("EMBEDDING_DIM")),
                 ENCODER_INPUT_DIM,
                 std.parseInt(std.extVar("NUM_ENCODER_LAYERS")),
                 std.parseInt(std.extVar("CLF_HIDDEN_DIM")),
                 std.extVar("AGGREGATIONS"),
                 std.parseInt(std.extVar("ADD_ELMO")),
                 std.parseInt(std.extVar("ADD_VAE")),
                 std.parseInt(std.extVar("ADD_BERT")))
    else if std.extVar("CLASSIFIER") == "cnn" then
        CNN_CLF(std.parseInt(std.extVar("EMBEDDING_DIM")),
                ENCODER_INPUT_DIM,
                std.parseInt(std.extVar("NUM_FILTERS")),
                std.parseInt(std.extVar("CLF_HIDDEN_DIM")),
                std.parseInt(std.extVar("ADD_ELMO")),
                std.parseInt(std.extVar("ADD_VAE")),
                std.parseInt(std.extVar("ADD_BERT")))

    else if std.extVar("CLASSIFIER") == "boe" then
        BOE_CLF(std.parseInt(std.extVar("EMBEDDING_DIM")),
                ENCODER_INPUT_DIM,
                std.parseInt(std.extVar("ADD_ELMO")),
                std.parseInt(std.extVar("ADD_VAE")),
                std.parseInt(std.extVar("ADD_BERT")))
    else if std.extVar("CLASSIFIER") == 'lr' then
        LR_CLF(std.parseInt(std.extVar("ADD_VAE")));

{
   "numpy_seed": std.extVar("SEED"),
   "pytorch_seed": std.extVar("SEED"),
   "random_seed": std.extVar("SEED"),
   "dataset_reader": BASE_READER(std.parseInt(std.extVar("ADD_ELMO")), std.parseInt(std.extVar("ADD_VAE")), std.parseInt(std.extVar("ADD_BERT")), std.extVar("THROTTLE"), std.parseInt(std.extVar("USE_SPACY_TOKENIZER"))),
    "validation_dataset_reader": BASE_READER(std.parseInt(std.extVar("ADD_ELMO")), std.parseInt(std.extVar("ADD_VAE")), std.parseInt(std.extVar("ADD_BERT")), null, std.parseInt(std.extVar("USE_SPACY_TOKENIZER"))),
   "datasets_for_vocab_creation": ["train"],
   "train_data_path": std.extVar("TRAIN_PATH"),
   "validation_data_path": std.extVar("DEV_PATH"),
   "model": {"type": "classifier"} + CLASSIFIER,
    "iterator": {
      "batch_size": std.parseInt(std.extVar("BATCH_SIZE")),
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
} + if std.parseInt(std.extVar("ADD_VAE")) == 1 then VOCABULARY_WITH_VAE else {}
