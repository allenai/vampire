local CUDA_DEVICE = std.parseInt(std.extVar("CUDA_DEVICE"));

local WORD_FILTER = {
    "word_filter": {
      "type": "regex_and_stopwords",
      "patterns": [
        "\\w{1,3}\\b", // tokens of length <= 3
        "\\w*\\d+\\w*", // words that contain digits,
         "\\w*[^\\P{P}]+\\w*" // punctuation
      ],
      "tokens_to_add": [">", "<", "$", "href=", "|", "°", "+", "£"],
      "stopword_file": std.extVar("STOPWORDS_PATH")
    }
};

local GLOVE_FIELDS(trainable) = {
  "glove_indexer": {
    "tokens": {
      "type": "single_id",
      "namespace": "vae",
      "lowercase_tokens": true,
    }
  },
  "glove_embedder": {
    "tokens": {
        "embedding_dim": 50,
        "trainable": trainable,
        "vocab_namespace": "vae",
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.50d.txt.gz",
    }
  },
  "embedding_dim": 50
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
      "archive_file": "s3://suching-dev/pretrained-models/elmo/imdb/model.tar.gz",
      // "archive_file": "s3://allennlp/models/transformer-elmo-2019.01.10.tar.gz",
      "dropout": 0.0,
      "bos_eos_tokens": ["<S>", "</S>"],
      "remove_bos_eos": true,
      "requires_grad": trainable
    }
  },
  "embedding_dim": 1024
};


local GLOVE_TRAINABLE = false;
local GLOVE_TOKEN_INDEXER = GLOVE_FIELDS(GLOVE_TRAINABLE)['glove_indexer'];
local GLOVE_TOKEN_EMBEDDER = GLOVE_FIELDS(GLOVE_TRAINABLE)['glove_embedder'];
local GLOVE_EMBEDDING_DIM = GLOVE_FIELDS(GLOVE_TRAINABLE)['embedding_dim'];



local BOE_FIELDS(embedding_dim, averaged) = {
    "type": "seq2vec",
    "architecture": {
        "embedding_dim": embedding_dim,
        "type": "boe",
        "averaged": averaged
    }
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




// local ENCODER = if std.extVar("ADDITIONAL_ENCODER") == "AVERAGE" then BOE_FIELDS(GLOVE_EMBEDDING_DIM, true) else {} +
//                 if std.extVar("ADDITIONAL_ENCODER") == "CNN" then CNN_FIELDS(std.parseInt(std.extVar("MAX_FILTER_SIZE")),
//                                                                   GLOVE_EMBEDDING_DIM,
//                                                                   std.parseInt(std.extVar("HIDDEN_SIZE")),
//                                                                   std.extVar("NUM_FILTERS")) else {};


local BASE_READER(THROTTLE, ADDITIONAL_UNLABELED_DATA_PATH, USE_SPACY_TOKENIZER, SEQUENCE_LENGTH, LAZY) = {
  "lazy": LAZY == 1,
  "type": "semisupervised_text_classification_json",
  "tokenizer": {
    "word_splitter": if USE_SPACY_TOKENIZER == 1 then "spacy" else "just_spaces",
    
  } + WORD_FILTER,
  "token_indexers": {
    "tokens": {
      "type": "single_id",
      "namespace": "vae",
      "lowercase_tokens": true
    } 
  } + GLOVE_TOKEN_INDEXER,
  "additional_unlabeled_data_path": ADDITIONAL_UNLABELED_DATA_PATH,
  "max_sequence_length": SEQUENCE_LENGTH,
  "ignore_labels": true,
  "sample": THROTTLE
};

{
   "numpy_seed": std.extVar("SEED"),
   "pytorch_seed": std.extVar("SEED"),
   "random_seed": std.extVar("SEED"),
   "dataset_reader": BASE_READER(std.extVar("THROTTLE"), std.extVar("ADDITIONAL_UNLABELED_DATA_PATH"), std.parseInt(std.extVar("USE_SPACY_TOKENIZER")), std.parseInt(std.extVar("SEQUENCE_LENGTH")), std.parseInt(std.extVar("LAZY_DATASET_READER"))),
   "validation_dataset_reader": BASE_READER(null, null, std.parseInt(std.extVar("USE_SPACY_TOKENIZER")), std.parseInt(std.extVar("SEQUENCE_LENGTH")), std.parseInt(std.extVar("LAZY_DATASET_READER"))),
   "datasets_for_vocab_creation": [
      "train"
   ],
   "train_data_path": std.extVar("TRAIN_PATH"),
   "validation_data_path": std.extVar("DEV_PATH"),
   "vocabulary": {
      "max_vocab_size": {
         "vae": std.parseInt(std.extVar("VOCAB_SIZE"))
      },
    //   "directory_path":  "/home/suching/vampire/vocabulary",
      "type": "extended_vocabulary"
   },
   "model": {
      "type": "vampire",
    //   "num_sources": std.parseInt(std.extVar("NUM_SOURCES")),
      "apply_batchnorm": std.parseInt(std.extVar("APPLY_BATCHNORM")) == 1,
      "bow_embedder": {
         "type": "bag_of_word_counts",
         "vocab_namespace": "vae",
         "ignore_oov": true
      },
    //   "additional_input_embedder": {
    //       "token_embedders": GLOVE_TOKEN_EMBEDDER
    //   },
    //   "additional_input_encoder": ENCODER,
      "kl_weight_annealing": std.extVar("KL_ANNEALING"),
      "sigmoid_weight_1": std.extVar("SIGMOID_WEIGHT_1"),
      "sigmoid_weight_2": std.extVar("SIGMOID_WEIGHT_2"),
      "linear_scaling": std.extVar("LINEAR_SCALING"),
      "reference_counts": std.extVar("REFERENCE_COUNTS"),
      "reference_vocabulary": std.extVar("REFERENCE_VOCAB"),
      "update_background_freq": std.parseInt(std.extVar("UPDATE_BACKGROUND_FREQUENCY")) == 1,
      "track_npmi": std.parseInt(std.extVar("TRACK_NPMI")) == 1,
      "vae": {
         "z_dropout": std.extVar("Z_DROPOUT"),
         "apply_batchnorm": std.parseInt(std.extVar("APPLY_BATCHNORM_1")) == 1,
         "encoder": {
            "activations": std.makeArray(std.parseInt(std.extVar("NUM_ENCODER_LAYERS")), function(i) std.extVar("ENCODER_ACTIVATION")),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_ENCODER_LAYERS")), function(i) std.parseInt(std.extVar("VAE_HIDDEN_DIM"))),
            "input_dim": std.parseInt(std.extVar("VOCAB_SIZE")) + 2,
            // "input_dim": GLOVE_EMBEDDING_DIM + std.parseInt(std.extVar("NUM_SOURCES")),
            "num_layers": std.parseInt(std.extVar("NUM_ENCODER_LAYERS"))
         },
         "mean_projection": {
            "activations": std.extVar("MEAN_PROJECTION_ACTIVATION"),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_MEAN_PROJECTION_LAYERS")), function(i) std.parseInt(std.extVar("VAE_HIDDEN_DIM"))),
            "input_dim": std.extVar("VAE_HIDDEN_DIM"),
            "num_layers": std.parseInt(std.extVar("NUM_MEAN_PROJECTION_LAYERS"))
         },
        "log_variance_projection": {
            "activations": std.extVar("LOG_VAR_PROJECTION_ACTIVATION"),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_LOG_VAR_PROJECTION_LAYERS")), function(i) std.parseInt(std.extVar("VAE_HIDDEN_DIM"))),
            "input_dim": std.parseInt(std.extVar("VAE_HIDDEN_DIM")),
            "num_layers": std.parseInt(std.extVar("NUM_LOG_VAR_PROJECTION_LAYERS"))
         },
         "decoder": {
            "activations": std.extVar("DECODER_ACTIVATION"),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("DECODER_NUM_LAYERS")), function(i) std.parseInt(std.extVar("DECODER_HIDDEN_DIM"))) + [std.parseInt(std.extVar("VOCAB_SIZE")) + 2],
            "input_dim": std.parseInt(std.extVar("VAE_HIDDEN_DIM")),
            "num_layers": std.parseInt(std.extVar("DECODER_NUM_LAYERS")) + 1
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
      "num_serialized_models_to_keep": 1,
      "num_epochs": 200,
      "patience": 75,
      "optimizer": {
         "lr": std.extVar("LEARNING_RATE"),
         "type": "adam"
      },
      "validation_metric": std.extVar("VALIDATION_METRIC")
   }
}
