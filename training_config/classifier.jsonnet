local USE_LAZY_DATASET_READER = std.parseInt(std.extVar("LAZY_DATASET_READER")) == 1;

// GPU to use. Setting this to -1 will mean that we'll use the CPU.
local CUDA_DEVICE = std.parseInt(std.extVar("CUDA_DEVICE"));

// Paths to data.
local TRAIN_PATH = std.extVar("TRAIN_PATH");
local DEV_PATH =  std.extVar("DEV_PATH");
local TEST_PATH =  std.extVar("TEST_PATH");

// Throttle the training data to a random subset of this length.
local THROTTLE = std.extVar("THROTTLE");

// Use the SpaCy tokenizer when reading in the data. If this is false, we'll use the just_spaces tokenizer.
local USE_SPACY_TOKENIZER = std.parseInt(std.extVar("USE_SPACY_TOKENIZER"));

// learning rate of overall model.
local LEARNING_RATE = std.extVar("LEARNING_RATE");

// dropout applied after pooling
local DROPOUT = std.extVar("DROPOUT");

// dropout applied after embedding
local EMBEDDING_DROPOUT = std.extVar("EMBEDDING_DROPOUT");

// batch size of model
local BATCH_SIZE = std.parseInt(std.extVar("BATCH_SIZE"));

// space-separated list of embeddings to use
local EMBEDDINGS = std.split(std.extVar("EMBEDDINGS"), " ");

// space-separated list of embeddings to freeze
local FREEZE_EMBEDDINGS = std.split(std.extVar("FREEZE_EMBEDDINGS"), " ");

// whether or not to evaluate on test
local EVALUATE_ON_TEST = std.parseInt(std.extVar("EVALUATE_ON_TEST")) == 1;

// total number of epochs to train for
local NUM_EPOCHS = std.parseInt(std.extVar("NUM_EPOCHS"));


// ----------------------------
// ENCODERS FOR CLASSIFICATION
// ----------------------------

// Bag-of-embeddings encoder fields
local BOE_FIELDS(embedding_dim, averaged) = {
    "type": "seq2vec",
    "architecture": {
        "embedding_dim": embedding_dim,
        "type": "boe",
        "averaged": averaged
    }
};

// LSTM encoder fields
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

// CNN encoder fields
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

// Maxpooling fields
local MAXPOOL_FIELDS(embedding_dim) = {
    "type": "seq2vec",
    "architecture": {
        "type": "maxpool",
        "embedding_dim": embedding_dim
    }
};

// ----------------------------
// INPUT EMBEDDINGS
// ----------------------------

local GLOVE_FIELDS(trainable) = {
  "glove_indexer": {
    "tokens": {
      "type": "single_id",
      "lowercase_tokens": true,
    }
  },
  "glove_embedder": {
    "tokens": {
        "embedding_dim": 300,
        "trainable": trainable,
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
    }
  },
  "embedding_dim": 300
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

local ELMO_LSTM_FIELDS(trainable, dropout) = {
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
      "dropout": dropout,
      "requires_grad": trainable
    }
  },
  "embedding_dim": 1024
};

local VAMPIRE_FIELDS(trainable, dropout) = {
    "vampire_indexer": {
        "vampire_tokens": {
            "type": "single_id",
            "namespace": "vampire",
            "lowercase_tokens": true
        }
    },  
    "vampire_embedder": {
        "vampire_tokens": {
                "type": "vampire_token_embedder",
                "expand_dim": true,
                "requires_grad": trainable,
                "device": CUDA_DEVICE,
                "model_archive": std.extVar("VAMPIRE_DIR") + "/model.tar.gz",
                "background_frequency": std.extVar("DATA_DIR") + "/vampire.bgfreq",
                "dropout": dropout
        }
    },
    "vocabulary": {
        "vocabulary":{
                "type": "vocabulary_with_vampire",
              "vampire_vocab_file": std.extVar("VAMPIRE_DIR") + "/vocabulary/vampire.txt",
        }
    },
    "embedding_dim": std.parseInt(std.extVar("VAMPIRE_DIM"))
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
        "embedding_dim": 50,
        "trainable": trainable,
        "type": "embedding",
    }
  },
  "embedding_dim": 50
};

// ----------------------------------------
// ALLENNLP CONFIGURATION FROM FIELDS ABOVE
// ----------------------------------------

local RANDOM_TRAINABLE = if std.count(FREEZE_EMBEDDINGS, "RANDOM") > 0 == false then true else false;
local VAMPIRE_TRAINABLE = if std.count(FREEZE_EMBEDDINGS, "VAMPIRE") > 0 == false then true else false;
local ELMO_TRAINABLE = if std.count(FREEZE_EMBEDDINGS, "ELMO_LSTM") > 0 == false then true else false;
local BERT_TRAINABLE = if std.count(FREEZE_EMBEDDINGS, "BERT") > 0 == false then true else false;


local RANDOM_TOKEN_INDEXER = if std.count(EMBEDDINGS, "RANDOM") > 0 then RANDOM_FIELDS(RANDOM_TRAINABLE)['random_indexer'] else {};
local VAMPIRE_TOKEN_INDEXER = if std.count(EMBEDDINGS, "VAMPIRE") > 0 then VAMPIRE_FIELDS(VAMPIRE_TRAINABLE, EMBEDDING_DROPOUT)['vampire_indexer'] else {};
local ELMO_TOKEN_INDEXER = if std.count(EMBEDDINGS, "ELMO_LSTM") > 0 then ELMO_LSTM_FIELDS(ELMO_TRAINABLE, EMBEDDING_DROPOUT)['elmo_lstm_indexer'] else {};
local BERT_TOKEN_INDEXER = if std.count(EMBEDDINGS, "BERT") > 0 then BERT_FIELDS(BERT_TRAINABLE)['bert_indexer'] else {};

local TOKEN_INDEXERS = RANDOM_TOKEN_INDEXER + VAMPIRE_TOKEN_INDEXER + ELMO_TOKEN_INDEXER + BERT_TOKEN_INDEXER;

local RANDOM_TOKEN_EMBEDDER = if std.count(EMBEDDINGS, "RANDOM") > 0 then RANDOM_FIELDS(RANDOM_TRAINABLE)['random_embedder'] else {};
local VAMPIRE_TOKEN_EMBEDDER = if std.count(EMBEDDINGS, "VAMPIRE") > 0 then VAMPIRE_FIELDS(VAMPIRE_TRAINABLE, EMBEDDING_DROPOUT)['vampire_embedder'] else {};
local ELMO_TOKEN_EMBEDDER = if std.count(EMBEDDINGS, "ELMO_LSTM") > 0 then ELMO_LSTM_FIELDS(ELMO_TRAINABLE, EMBEDDING_DROPOUT)['elmo_lstm_embedder'] else {};
local BERT_TOKEN_EMBEDDER = if std.count(EMBEDDINGS, "BERT") > 0 then BERT_FIELDS(BERT_TRAINABLE)['bert_embedder'] else {};

local TOKEN_EMBEDDERS = RANDOM_TOKEN_EMBEDDER + VAMPIRE_TOKEN_EMBEDDER + ELMO_TOKEN_EMBEDDER + BERT_TOKEN_EMBEDDER;

local RANDOM_EMBEDDING_DIM = if std.count(EMBEDDINGS, "RANDOM") > 0 then RANDOM_FIELDS(RANDOM_TRAINABLE)['embedding_dim'] else 0;
local VAMPIRE_EMBEDDING_DIM = if std.count(EMBEDDINGS, "VAMPIRE") > 0 then VAMPIRE_FIELDS(VAMPIRE_TRAINABLE, EMBEDDING_DROPOUT)['embedding_dim'] else 0;
local ELMO_EMBEDDING_DIM = if std.count(EMBEDDINGS, "ELMO_LSTM") > 0 then ELMO_LSTM_FIELDS(ELMO_TRAINABLE, EMBEDDING_DROPOUT)['embedding_dim'] else 0;
local BERT_EMBEDDING_DIM = if std.count(EMBEDDINGS, "BERT") > 0 then BERT_FIELDS(BERT_TRAINABLE)['embedding_dim'] else 0;

local EMBEDDING_DIM = RANDOM_EMBEDDING_DIM + VAMPIRE_EMBEDDING_DIM + ELMO_EMBEDDING_DIM + BERT_EMBEDDING_DIM;

local ENCODER = if std.extVar("ENCODER") == "AVERAGE" then BOE_FIELDS(EMBEDDING_DIM, true) else {} + 
                if std.extVar("ENCODER") == "SUM" then BOE_FIELDS(EMBEDDING_DIM, false) else {} + 
                if std.extVar("ENCODER") == "MAXPOOL" then MAXPOOL_FIELDS(EMBEDDING_DIM) else {} + 
                if std.extVar("ENCODER") == "LSTM" then LSTM_FIELDS(std.parseInt(std.extVar("NUM_ENCODER_LAYERS")),
                                                                    EMBEDDING_DIM,
                                                                    std.parseInt(std.extVar("HIDDEN_SIZE")),
                                                                    std.extVar("AGGREGATIONS")) else {} +
                if std.extVar("ENCODER") == "CNN" then CNN_FIELDS(std.parseInt(std.extVar("MAX_FILTER_SIZE")),
                                                                  EMBEDDING_DIM,
                                                                  std.parseInt(std.extVar("HIDDEN_SIZE")),
                                                                  std.extVar("NUM_FILTERS")) else {};

// ---------------------
// BASE DATASET READER
// ---------------------

local BASE_READER(TOKEN_INDEXERS, THROTTLE, USE_LAZY_DATASET_READER) = {
    "lazy": USE_LAZY_DATASET_READER,
    "type": "semisupervised_text_classification_json",
    "token_indexers": TOKEN_INDEXERS,
    "max_sequence_length": 400,
    "sample": THROTTLE,
};


// ----------------------------
// ALLENNLP CONFIG
// ----------------------------


{
   "numpy_seed": std.extVar("SEED"),
   "pytorch_seed": std.extVar("SEED"),
   "random_seed": std.extVar("SEED"),
   "dataset_reader": BASE_READER(TOKEN_INDEXERS, THROTTLE, USE_LAZY_DATASET_READER),
   "validation_dataset_reader": BASE_READER(TOKEN_INDEXERS, null, USE_LAZY_DATASET_READER),
   "datasets_for_vocab_creation": ["train"],
   "train_data_path": TRAIN_PATH,
   "validation_data_path": DEV_PATH,
   "test_data_path": if EVALUATE_ON_TEST then TEST_PATH else null,
   "evaluate_on_test": EVALUATE_ON_TEST,
   "model": {
      "type": "classifier",
      "input_embedder": {
                "token_embedders": TOKEN_EMBEDDERS
      } + if std.count(EMBEDDINGS, "BERT") > 0  then BERT_FIELDS(BERT_TRAINABLE)['extra_embedder_fields'] else {},
      "encoder": ENCODER,
      "dropout": DROPOUT
   },	
   "data_loader": {
        "batch_sampler": {
            "type": "basic",
            "sampler": "sequential",
            "batch_size": std.parseInt(std.extVar("BATCH_SIZE")),
            "drop_last": false
        }
    },
   "trainer": {
      "cuda_device": CUDA_DEVICE,
      "num_epochs": NUM_EPOCHS,
      "optimizer": {
         "lr": LEARNING_RATE,
         "type": "adam_str_lr"
      },
      "patience": 5,
      "validation_metric": "+accuracy"
   }
} + if std.count(EMBEDDINGS, "VAMPIRE") > 0 then VAMPIRE_FIELDS(VAMPIRE_TRAINABLE, EMBEDDING_DROPOUT)['vocabulary'] else {}
