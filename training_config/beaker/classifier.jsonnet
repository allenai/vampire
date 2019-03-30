local USE_LAZY_DATASET_READER = std.parseInt(std.extVar("LAZY_DATASET_READER")) == 1;

// GPU to use. Setting this to -1 will mean that we'll use the CPU.
local CUDA_DEVICE = std.parseInt(std.extVar("CUDA_DEVICE"));

// Paths to data.
local TRAIN_PATH = std.extVar("DATA_DIR") + "train.jsonl";
local DEV_PATH =  std.extVar("DATA_DIR") + "dev.jsonl";
local TEST_PATH =  std.extVar("DATA_DIR") + "test.jsonl";

// Throttle the training data to a random subset of this length.
local THROTTLE = std.extVar("THROTTLE");

// Use the SpaCy tokenizer when reading in the data. If this is false, we'll use the just_spaces tokenizer.
local USE_SPACY_TOKENIZER = std.parseInt(std.extVar("USE_SPACY_TOKENIZER"));

// learning rate of overall model.
local LEARNING_RATE = std.extVar("LEARNING_RATE");

// dropout applied after pooling
local DROPOUT = std.parseInt(std.extVar("DROPOUT")) / 10;

local BATCH_SIZE = std.parseInt(std.extVar("BATCH_SIZE"));

local EMBEDDINGS = std.split(std.extVar("EMBEDDINGS"), " ");
local FREEZE_EMBEDDINGS = std.split(std.extVar("FREEZE_EMBEDDINGS"), " ");

local EVALUATE_ON_TEST = std.parseInt(std.extVar("EVALUATE_ON_TEST")) == 1;

local NUM_EPOCHS = std.parseInt(std.extVar("NUM_EPOCHS"));

local BOE_FIELDS(embedding_dim, averaged) = {
    "type": "seq2vec",
    "architecture": {
        "embedding_dim": embedding_dim,
        "type": "boe",
        "averaged": averaged
    }
};

local LSTM_FIELDS(num_encoder_layers, embedding_dim, hidden_size, aggregations) = {
      "type" : "seq2seq",
      "architecture": {
        "type": "lstm",
        "num_layers": num_encoder_layers,
        "bidirectional": true,
        "input_size": embedding_dim,
        "hidden_size": hidden_size
      },
      "aggregations": std.split(aggregations, ",")
};

local CNN_FIELDS(max_filter_size, embedding_dim, hidden_size, num_filters) = {
      "type": "seq2vec",
      "architecture": {
          "type": "cnn",
          "ngram_filter_sizes": std.range(1, max_filter_size),
          "num_filters": num_filters,
          "embedding_dim": embedding_dim,
          "output_dim": hidden_size, 
      },
};

local MAXPOOL_FIELDS(embedding_dim) = {
    "type": "seq2vec",
    "architecture": {
        "type": "maxpool",
        "embedding_dim": embedding_dim
    }
};

local CLS_TOKEN_FIELDS(embedding_dim) = {
    "type": "seq2vec",
    "architecture": {
        "type": "cls_token",
        "embedding_dim": embedding_dim
    }
};


local ELMO_LSTM_FIELDS(trainable) = {
  "elmo_lstm_indexer": {
    "elmo": {
      "type": "elmo_characters",
    }
  },
  "elmo_lstm_embedder": {
    "elmo": {
      "type": "elmo_token_embedder",
      "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
      "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
      "do_layer_norm": false,
      "dropout": 0.0,
      "requires_grad": trainable
    }
  },
  "embedding_dim": 1024
};

local BERT_FIELDS(trainable) = {
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
        "requires_grad": trainable,
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
  "embedding_dim": 768
};

local ELMO_TRANSFORMER_FIELDS(trainable) = {
  "elmo_transformer_indexer": {
    "elmo": {
      "type": "elmo_characters",
    }
  },
  "elmo_transformer_embedder": {
    "elmo": {
      "type": "bidirectional_lm_token_embedder",
      "archive_file": "s3://allennlp/models/transformer-elmo-2019.01.10.tar.gz",
      "dropout": 0.0,
      "bos_eos_tokens": ["<S>", "</S>"],
      "remove_bos_eos": true,
      "requires_grad": trainable
    }
  },
  "embedding_dim": 1024
};

local GLOVE_FIELDS(trainable) = {
  "glove_indexer": {
    "tokens": {
      "type": "single_id",
      "lowercase_tokens": true,
    }
  },
  "glove_embedder": {
    "tokens": {
        "embedding_dim": 50,
        "trainable": trainable,
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.50d.txt.gz",
    }
  },
  "embedding_dim": 50
};

local W2V_FIELDS(trainable) = {
  "w2v_indexer": {
    "tokens": {
      "type": "single_id",
      "lowercase_tokens": false,
    }
  },
  "w2v_embedder": {
    "tokens": {
        "embedding_dim": 300,
        "trainable": trainable,
        "pretrained_file": "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip",
    }
  },
  "embedding_dim": 300
};

local VAMPIRE_FIELDS(trainable) = {
    "vampire_indexer": {
        "vampire_tokens": {
            "type": "single_id",
            "namespace": "vae",
            "lowercase_tokens": true
        }
    },  
    "vampire_embedder": {
        "vampire_tokens": {
                "type": "vampire_token_embedder",
                "expand_dim": true,
                "requires_grad": trainable,
                "model_archive": "/home/sg/Github/vampire/model_logs/pretrained_vampire/model.tar.gz",
                "background_frequency": "/home/sg/Github/vampire/model_logs/pretrained_vampire/vocabulary/vae.bgfreq.json",
                "dropout": 0.0
        }
    },
    "vocabulary": {
        "vocabulary":{
              "type": "vocabulary_with_vae",
              "vae_vocab_file": "/home/sg/Github/vampire/model_logs/pretrained_vampire/vocabulary/vae.txt",
        }
    },
    "embedding_dim": 64
};



local CHARACTER_ENCODER = if std.count(EMBEDDINGS, "CHAR_AVERAGE") > 0 then BOE_FIELDS(std.parseInt(std.extVar("CHARACTER_EMBEDDING_DIM")), true) else {} + 
                          if std.count(EMBEDDINGS, "CHAR_LSTM") > 0 then LSTM_FIELDS(std.parseInt(std.extVar("NUM_CHARACTER_ENCODER_LAYERS")), std.parseInt(std.extVar("CHARACTER_EMBEDDING_DIM")), std.parseInt(std.extVar("CHARACTER_HIDDEN_SIZE")), null) else {} +
                          if std.count(EMBEDDINGS, "CHAR_CNN") > 0 then CNN_FIELDS(std.parseInt(std.extVar("MAX_CHARACTER_FILTER_SIZE")), std.parseInt(std.extVar("CHARACTER_EMBEDDING_DIM")), std.parseInt(std.extVar("CHARACTER_HIDDEN_SIZE")), std.parseInt(std.extVar("NUM_CHARACTER_FILTERS"))) else {};


local CHAR_AVERAGE_FIELDS(trainable) = {
  "boe_indexer": {
    "token_characters": {
      "type": "characters",
    }
  },
  "boe_embedder": {
    "token_characters": {
      "type": "character_encoding",
        "embedding": {
          "trainable": trainable,
          "embedding_dim": std.parseInt(std.extVar("CHARACTER_EMBEDDING_DIM"))
      },
      "encoder": {"type": "boe"} + CHARACTER_ENCODER['architecture'],
    }
  },
  "embedding_dim": std.parseInt(std.extVar("CHARACTER_EMBEDDING_DIM"))
};

local CHAR_CNN_FIELDS(trainable) = {
  "cnn_indexer": {
    "token_characters": {
      "type": "characters",
      "min_padding_length": "5",      
    }
  },
  "cnn_embedder": {
    "token_characters": {
      "type": "character_encoding",
        "embedding": {
          "trainable": trainable,
          "embedding_dim": std.parseInt(std.extVar("CHARACTER_EMBEDDING_DIM"))
      },
      "encoder": {"type": "cnn"} + CHARACTER_ENCODER['architecture'],
    }
  },
  "embedding_dim": std.parseInt(std.extVar("CHARACTER_HIDDEN_SIZE"))
};

local CHAR_LSTM_FIELDS(trainable) = {
  "lstm_indexer": {
    "token_characters": {
      "type": "characters",
      "min_padding_length": "5",      
    }
  },
  "lstm_embedder": {
    "token_characters": {
      "type": "character_encoding",
        "embedding": {
          "trainable": trainable,
          "embedding_dim": std.parseInt(std.extVar("CHARACTER_EMBEDDING_DIM"))
      },
      "encoder": {"type": "lstm"} + CHARACTER_ENCODER['architecture'],
    }
  },
  "embedding_dim": std.parseInt(std.extVar("CHARACTER_HIDDEN_SIZE")) * 2
};

local RANDOM_FIELDS(trainable) = {
  "random_indexer": {
    "tokens": {
      "type": "single_id",
      "lowercase_tokens": true,
    }
  },
  "random_embedder": {
    "tokens": {
        "embedding_dim": 300,
        "trainable": trainable,
        "type": "embedding",
    }
  },
  "embedding_dim": 300
};


local BOW_COUNT_FIELDS = {
  "bow_indexer": {
    "tokens": {
      "type": "single_id",
      "lowercase_tokens": true,
    }
  },
  "bow_embedder": {
    "tokens": {
        "type": "bag_of_word_counts",
    }
  },
  "embedding_dim": 0
};

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



local ELMO_LSTM_TRAINABLE = if std.count(FREEZE_EMBEDDINGS, "ELMO_LSTM") > 0 == false then true else false;
local BERT_TRAINABLE = if std.count(FREEZE_EMBEDDINGS, "BERT") > 0 == false then true  else false;
local RANDOM_TRAINABLE = if std.count(FREEZE_EMBEDDINGS, "RANDOM") > 0 == false then true  else false;
local W2V_TRAINABLE = if std.count(FREEZE_EMBEDDINGS, "W2V") > 0 == false then true  else false;
local CHAR_CNN_TRAINABLE = if std.count(FREEZE_EMBEDDINGS, "CHAR_CNN") > 0 == false then true  else false;
local CHAR_LSTM_TRAINABLE = if std.count(FREEZE_EMBEDDINGS, "CHAR_LSTM") > 0 == false then true  else false;
local CHAR_AVERAGE_TRAINABLE = if std.count(FREEZE_EMBEDDINGS, "CHAR_AVERAGE") > 0 == false then true  else false;
local GLOVE_TRAINABLE = if std.count(FREEZE_EMBEDDINGS, "GLOVE") > 0 == false then true  else false;
local ELMO_TRANSFORMER_TRAINABLE = if std.count(FREEZE_EMBEDDINGS, "ELMO_TRANSFORMER") > 0 == false then true  else false;
local VAMPIRE_TRAINABLE = if std.count(FREEZE_EMBEDDINGS, "VAMPIRE") > 0 == false then true else false;

local RANDOM_TOKEN_INDEXER = if std.count(EMBEDDINGS, "RANDOM") > 0 then RANDOM_FIELDS(RANDOM_TRAINABLE)['random_indexer'] else {};

local VAMPIRE_TOKEN_INDEXER = if std.count(EMBEDDINGS, "VAMPIRE") > 0 then VAMPIRE_FIELDS(VAMPIRE_TRAINABLE)['vampire_indexer'] else {};


local ELMO_LSTM_TOKEN_INDEXER = if std.count(EMBEDDINGS, "ELMO_LSTM") > 0 then ELMO_LSTM_FIELDS(ELMO_LSTM_TRAINABLE)['elmo_lstm_indexer'] else {};
local BERT_TOKEN_INDEXER = if std.count(EMBEDDINGS, "BERT") > 0 then BERT_FIELDS(BERT_TRAINABLE)['bert_indexer'] else {};
local RANDOM_TOKEN_INDEXER = if std.count(EMBEDDINGS, "RANDOM") > 0 then RANDOM_FIELDS(RANDOM_TRAINABLE)['random_indexer'] else {};
local W2V_TOKEN_INDEXER = if std.count(EMBEDDINGS, "W2V") > 0 then W2V_FIELDS(W2V_TRAINABLE)['w2v_indexer'] else {};
local CHAR_CNN_TOKEN_INDEXER = if std.count(EMBEDDINGS, "CHAR_CNN") > 0 then CHAR_CNN_FIELDS(CHAR_CNN_TRAINABLE)['cnn_indexer'] else {};
local CHAR_LSTM_TOKEN_INDEXER = if std.count(EMBEDDINGS, "CHAR_LSTM") > 0 then CHAR_LSTM_FIELDS(CHAR_LSTM_TRAINABLE)['lstm_indexer'] else {};
local CHAR_AVERAGE_TOKEN_INDEXER = if std.count(EMBEDDINGS, "CHAR_AVERAGE") > 0 then CHAR_AVERAGE_FIELDS(CHAR_AVERAGE_TRAINABLE)['boe_indexer'] else {};
local GLOVE_TOKEN_INDEXER = if std.count(EMBEDDINGS, "GLOVE") > 0 then GLOVE_FIELDS(GLOVE_TRAINABLE)['glove_indexer'] else {};
local ELMO_TRANSFORMER_TOKEN_INDEXER = if std.count(EMBEDDINGS, "ELMO_TRANSFORMER") > 0 then ELMO_TRANSFORMER_FIELDS(ELMO_TRANSFORMER_TRAINABLE)['elmo_transformer_indexer'] else {};
local VAMPIRE_TOKEN_INDEXER = if std.count(EMBEDDINGS, "VAMPIRE") > 0 then VAMPIRE_FIELDS(VAMPIRE_TRAINABLE)['vampire_indexer'] else {};

local TOKEN_INDEXERS = ELMO_LSTM_TOKEN_INDEXER
                       + BERT_TOKEN_INDEXER
                       + RANDOM_TOKEN_INDEXER
                       + W2V_TOKEN_INDEXER 
                       + CHAR_CNN_TOKEN_INDEXER
                       + CHAR_LSTM_TOKEN_INDEXER
                       + CHAR_AVERAGE_TOKEN_INDEXER
                       + GLOVE_TOKEN_INDEXER
                       + ELMO_TRANSFORMER_TOKEN_INDEXER
                       + VAMPIRE_TOKEN_INDEXER;

local ELMO_LSTM_TOKEN_EMBEDDER = if std.count(EMBEDDINGS, "ELMO_LSTM") > 0 then ELMO_LSTM_FIELDS(ELMO_LSTM_TRAINABLE)['elmo_lstm_embedder'] else {};
local BERT_TOKEN_EMBEDDER = if std.count(EMBEDDINGS, "BERT") > 0 then BERT_FIELDS(BERT_TRAINABLE)['bert_embedder'] else {};
local RANDOM_TOKEN_EMBEDDER = if std.count(EMBEDDINGS, "RANDOM") > 0 then RANDOM_FIELDS(RANDOM_TRAINABLE)['random_embedder'] else {};
local W2V_TOKEN_EMBEDDER = if std.count(EMBEDDINGS, "W2V") > 0 then W2V_FIELDS(W2V_TRAINABLE)['w2v_embedder'] else {};
local CHAR_CNN_TOKEN_EMBEDDER = if std.count(EMBEDDINGS, "CHAR_CNN") > 0 then CHAR_CNN_FIELDS(CHAR_CNN_TRAINABLE)['cnn_embedder'] else {};
local CHAR_LSTM_TOKEN_EMBEDDER = if std.count(EMBEDDINGS, "CHAR_LSTM") > 0 then CHAR_LSTM_FIELDS(CHAR_LSTM_TRAINABLE)['lstm_embedder'] else {};
local CHAR_AVERAGE_TOKEN_EMBEDDER = if std.count(EMBEDDINGS, "CHAR_AVERAGE") > 0 then CHAR_AVERAGE_FIELDS(CHAR_AVERAGE_TRAINABLE)['boe_embedder'] else {};
local GLOVE_TOKEN_EMBEDDER = if std.count(EMBEDDINGS, "GLOVE") > 0 then GLOVE_FIELDS(GLOVE_TRAINABLE)['glove_embedder'] else {};
local ELMO_TRANSFORMER_TOKEN_EMBEDDER = if std.count(EMBEDDINGS, "ELMO_TRANSFORMER") > 0 then ELMO_TRANSFORMER_FIELDS(ELMO_TRANSFORMER_TRAINABLE)['elmo_transformer_embedder'] else {};
local VAMPIRE_TOKEN_EMBEDDER = if std.count(EMBEDDINGS, "VAMPIRE") > 0 then VAMPIRE_FIELDS(VAMPIRE_TRAINABLE)['vampire_embedder'] else {};

local TOKEN_EMBEDDERS = ELMO_LSTM_TOKEN_EMBEDDER
                       + BERT_TOKEN_EMBEDDER
                       + RANDOM_TOKEN_EMBEDDER
                       + W2V_TOKEN_EMBEDDER 
                       + CHAR_CNN_TOKEN_EMBEDDER
                       + CHAR_LSTM_TOKEN_EMBEDDER
                       + CHAR_AVERAGE_TOKEN_EMBEDDER
                       + GLOVE_TOKEN_EMBEDDER
                       + ELMO_TRANSFORMER_TOKEN_EMBEDDER
                       + VAMPIRE_TOKEN_EMBEDDER;

local ELMO_LSTM_EMBEDDING_DIM = if std.count(EMBEDDINGS, "ELMO_LSTM") > 0 then ELMO_LSTM_FIELDS(ELMO_LSTM_TRAINABLE)['embedding_dim'] else 0;
local BERT_EMBEDDING_DIM = if std.count(EMBEDDINGS, "BERT") > 0 then BERT_FIELDS(BERT_TRAINABLE)['embedding_dim'] else 0;
local RANDOM_EMBEDDING_DIM = if std.count(EMBEDDINGS, "RANDOM") > 0 then RANDOM_FIELDS(RANDOM_TRAINABLE)['embedding_dim'] else 0;
local W2V_EMBEDDING_DIM = if std.count(EMBEDDINGS, "W2V") > 0 then W2V_FIELDS(W2V_TRAINABLE)['embedding_dim'] else 0;
local CHAR_CNN_EMBEDDING_DIM = if std.count(EMBEDDINGS, "CHAR_CNN") > 0 then CHAR_CNN_FIELDS(CHAR_CNN_TRAINABLE)['embedding_dim'] else 0;
local CHAR_LSTM_EMBEDDING_DIM = if std.count(EMBEDDINGS, "CHAR_LSTM") > 0 then CHAR_LSTM_FIELDS(CHAR_LSTM_TRAINABLE)['embedding_dim'] else 0;
local CHAR_AVERAGE_EMBEDDING_DIM = if std.count(EMBEDDINGS, "CHAR_AVERAGE") > 0 then CHAR_AVERAGE_FIELDS(CHAR_AVERAGE_TRAINABLE)['embedding_dim'] else 0;
local GLOVE_EMBEDDING_DIM = if std.count(EMBEDDINGS, "GLOVE") > 0 then GLOVE_FIELDS(GLOVE_TRAINABLE)['embedding_dim'] else 0;
local ELMO_TRANSFORMER_EMBEDDING_DIM = if std.count(EMBEDDINGS, "ELMO_TRANSFORMER") > 0 then ELMO_TRANSFORMER_FIELDS(ELMO_TRANSFORMER_TRAINABLE)['embedding_dim'] else 0;
local VAMPIRE_EMBEDDING_DIM = if std.count(EMBEDDINGS, "VAMPIRE") > 0 then VAMPIRE_FIELDS(VAMPIRE_TRAINABLE)['embedding_dim'] else 0;

local EMBEDDING_DIM = ELMO_LSTM_EMBEDDING_DIM
                      + BERT_EMBEDDING_DIM
                      + RANDOM_EMBEDDING_DIM
                      + W2V_EMBEDDING_DIM 
                      + CHAR_CNN_EMBEDDING_DIM
                      + CHAR_LSTM_EMBEDDING_DIM
                      + CHAR_AVERAGE_EMBEDDING_DIM
                      + GLOVE_EMBEDDING_DIM
                      + ELMO_TRANSFORMER_EMBEDDING_DIM
                      + VAMPIRE_EMBEDDING_DIM;

local ENCODER = if std.extVar("ENCODER") == "AVERAGE" then BOE_FIELDS(EMBEDDING_DIM, true) else {} + 
                if std.extVar("ENCODER") == "SUM" then BOE_FIELDS(EMBEDDING_DIM, false) else {} + 
                if std.extVar("ENCODER") == "MAXPOOL" then MAXPOOL_FIELDS(EMBEDDING_DIM) else {} + 
                if std.extVar("ENCODER") == "CLS_TOKEN" then CLS_TOKEN_FIELDS(EMBEDDING_DIM) else {} + 
                if std.extVar("ENCODER") == "LSTM" then LSTM_FIELDS(std.parseInt(std.extVar("NUM_ENCODER_LAYERS")), EMBEDDING_DIM, std.parseInt(std.extVar("HIDDEN_SIZE")), std.extVar("AGGREGATIONS")) else {} +
                if std.extVar("ENCODER") == "CNN" then CNN_FIELDS(std.parseInt(std.extVar("MAX_FILTER_SIZE")), EMBEDDING_DIM, std.parseInt(std.extVar("HIDDEN_SIZE")), std.extVar("NUM_FILTERS")) else {};

local OUTPUT_LAYER_DIM = if std.extVar("ENCODER") == "AVERAGE" then EMBEDDING_DIM  else 0 + 
                         if std.extVar("ENCODER") == "SUM" then EMBEDDING_DIM  else 0 + 
                         if std.extVar("ENCODER") == "MAXPOOL" then EMBEDDING_DIM else 0 + 
                         if std.extVar("ENCODER") == "CLS_TOKEN" then EMBEDDING_DIM else 0 + 
                         if std.extVar("ENCODER") == "LSTM" then std.parseInt(std.extVar("HIDDEN_SIZE")) * 2 * std.length(std.split(std.extVar("AGGREGATIONS"), ",")) else 0 +
                         if std.extVar("ENCODER") == "CNN" then std.parseInt(std.extVar("HIDDEN_SIZE")) else 0;


local OUTPUT_LAYER_HIDDEN_DIM = if std.parseInt(std.extVar("NUM_OUTPUT_LAYERS")) == 1 then [OUTPUT_LAYER_DIM] else [] + 
                                if std.parseInt(std.extVar("NUM_OUTPUT_LAYERS")) == 2 then [OUTPUT_LAYER_DIM, OUTPUT_LAYER_DIM] else [] + 
                                if std.parseInt(std.extVar("NUM_OUTPUT_LAYERS")) == 3 then [OUTPUT_LAYER_DIM, OUTPUT_LAYER_DIM, OUTPUT_LAYER_DIM] else [];

local BASE_READER(TOKEN_INDEXERS, THROTTLE, USE_SPACY_TOKENIZER, USE_LAZY_DATASET_READER) = {
  "lazy": USE_LAZY_DATASET_READER,
  "type": "semisupervised_text_classification_json",
  "tokenizer": {
    "word_splitter": if USE_SPACY_TOKENIZER == 1 then "spacy" else "just_spaces",
  },
  "token_indexers": TOKEN_INDEXERS,
  "sample": THROTTLE,
};

{
   "numpy_seed": std.extVar("SEED"),
   "pytorch_seed": std.extVar("SEED"),
   "random_seed": std.extVar("SEED"),
   "dataset_reader": BASE_READER(TOKEN_INDEXERS, THROTTLE, USE_SPACY_TOKENIZER, USE_LAZY_DATASET_READER),
   "validation_dataset_reader": BASE_READER(TOKEN_INDEXERS, null, USE_SPACY_TOKENIZER, USE_LAZY_DATASET_READER),
   "datasets_for_vocab_creation": ["train"],
   "train_data_path": TRAIN_PATH,
   "validation_data_path": DEV_PATH,
   "test_data_path": if EVALUATE_ON_TEST then TEST_PATH else null,
   "evaluate_on_test": EVALUATE_ON_TEST,
   "model": {
      "type": "classifier",
      "input_embedder": {
                "token_embedders": TOKEN_EMBEDDERS
      } + if std.count(EMBEDDINGS, "BERT") > 0 then BERT_FIELDS['extra_embedder_fields'] else {},
      "encoder": ENCODER,
      "output_layer": {
        "input_dim": OUTPUT_LAYER_DIM,
        "num_layers": std.parseInt(std.extVar("NUM_OUTPUT_LAYERS")),
        "hidden_dims": OUTPUT_LAYER_HIDDEN_DIM,
        "activations": "relu",
        "dropout": DROPOUT
      },      
      "dropout": DROPOUT
   },	
    "iterator": {
      "batch_size": BATCH_SIZE,
      "type": "basic"
   },

   "trainer": {
      "cuda_device": CUDA_DEVICE,
      "num_epochs": NUM_EPOCHS,
      "optimizer": {
         "lr": LEARNING_RATE,
         "type": "adam"
      },
      "learning_rate_scheduler": {
          "type": "reduce_on_plateau",
          "factor": 0.5, 
          "patience": 2
      },
      "patience": 10,
      "num_serialized_models_to_keep": 1,
      "validation_metric": "+accuracy"
   }
} + if std.count(EMBEDDINGS, "VAMPIRE") > 0 then VAMPIRE_FIELDS(VAMPIRE_TRAINABLE)['vocabulary'] else {}
