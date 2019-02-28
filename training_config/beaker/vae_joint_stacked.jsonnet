local CUDA_DEVICE =
  if std.parseInt(std.extVar("NUM_GPU")) == 0 then
  -1
  else if std.parseInt(std.extVar("NUM_GPU")) > 1 then
  std.range(0, std.extVar("NUM_GPU") - 1)
  else if std.parseInt(std.extVar("NUM_GPU")) == 1 then
  0;

local ELMO_REQUIRES_GRAD = std.parseInt(std.extVar("ELMO_FINETUNE")) == 1;


local BERT_FIELDS = {
  "bert_indexer": {
     "bert": {
    "type": "bert-pretrained",
    "pretrained_model": std.extVar('BERT_VOCAB')
  }
  },
  "bert_embedder": {
  "bert": {
    "type": "bert-pretrained",
    "pretrained_model": std.extVar('BERT_WEIGHTS'),
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


local ELMO_FIELDS = {
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
      "requires_grad": ELMO_REQUIRES_GRAD
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
        "expand_dim": EXPAND_DIM,
        "requires_grad": std.parseInt(std.extVar("VAE_FINE_TUNE")) == 1,
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
local ELMO_INDEXER = if std.parseInt(std.extVar("ADD_ELMO")) == 1 then ELMO_FIELDS['elmo_indexer'] else {};
local BERT_INDEXER = if std.parseInt(std.extVar("ADD_BERT")) == 1 then BERT_FIELDS['bert_indexer'] else {};
local BASIC_INDEXER = if std.parseInt(std.extVar("ADD_BASIC")) == 1 then BASIC_FIELDS(std.parseInt(std.extVar("EMBEDDING_DIM")))['basic_indexer'] else {};
local GLOVE_INDEXER = if std.parseInt(std.extVar("ADD_GLOVE")) == 1 then GLOVE_FIELDS['glove_indexer'] else {};


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
  } + VAE_INDEXER + ELMO_INDEXER,
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
  } + VAE_INDEXER + ELMO_INDEXER,
  "sequence_length": 400,
  "sample": THROTTLE,
  "unlabeled_data_path": UNLABELED_DATA_PATH
};


local VAE_EMBEDDINGS = if std.parseInt(std.extVar("ADD_VAE")) == 1 then VAE_FIELDS(true)['vae_embedder'] else {};
local ELMO_EMBEDDINGS = if std.parseInt(std.extVar("ADD_ELMO")) == 1 then ELMO_FIELDS['elmo_embedder'] else {};
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

// Add the VAE vocab by default as the BOW embedder will project to pre-trained document representations.
VOCABULARY_WITH_VAE + {
  "numpy_seed": std.extVar("SEED"),
  "pytorch_seed": std.extVar("SEED"),
  "random_seed": std.extVar("SEED"),
  // "evaluate_on_test": std.parseInt(std.extVar("EVALUATE_ON_TEST")) == 1,
  "dataset_reader": BASE_JOINT_READER(false, std.extVar("THROTTLE"), std.extVar("UNLABELED_DATA_PATH"), std.parseInt(std.extVar("USE_SPACY_TOKENIZER")),
    std.parseInt(std.extVar("UNLABELED_DATA_FACTOR"))),
  "validation_dataset_reader": BASE_READER(false, null, null, std.parseInt(std.extVar("USE_SPACY_TOKENIZER"))),
  "datasets_for_vocab_creation": ["train"],
  "train_data_path": std.extVar("TRAIN_PATH"),
  "validation_data_path": std.extVar("DEV_PATH"),
  // "test_data_path": std.extVar("TEST_PATH"),
  "model": {
    "type": "joint_stacked_classifier",
    "classifier": CLASSIFIER + { "vae_embedding_dim": std.parseInt(std.extVar("M1_LATENT_DIM")), "dropout": std.parseInt(std.extVar("DROPOUT")) / 10.0 },
    "apply_batchnorm": true,
    "baseline_only": std.parseInt(std.extVar("BASELINE_ONLY")) == 1,

    "kl_weight_annealing": std.extVar("KL_ANNEALING"),
    "reference_counts": std.extVar("REFERENCE_COUNTS"),
    "reference_vocabulary": std.extVar("REFERENCE_VOCAB"),
    "update_background_freq": false,

    // Turn these off to prevent serialization of topics and NPMI tracking.
    "track_topics": false,
    "track_npmi": false,

    // The pre-trained VAE embeddings will enter via the BOW-embedder.
    // EXPAND_DIM = False will produce (batch, latent_dim) tensors.
    "bow_embedder":  VAE_FIELDS(false)['vae_embedder']['vae_tokens'],

    "alpha": std.parseInt(std.extVar("ALPHA")),

    // The joint model's VAE.    
    "vae": {
       "type": "logistic_normal",
       "apply_batchnorm": false,
       "encoder": {
          "activations": [ "softplus" for x in std.range(0, std.parseInt(std.extVar("NUM_VAE_ENCODER_LAYERS")) - 1) ],
          "input_dim": std.parseInt(std.extVar("M1_LATENT_DIM")) + std.parseInt(std.extVar("NUM_CLASSES")),
          "hidden_dims": [ std.parseInt(std.extVar("VAE_HIDDEN_DIM")) for x in std.range(0, std.parseInt(std.extVar("NUM_VAE_ENCODER_LAYERS")) - 1) ],
          "num_layers": std.parseInt(std.extVar("NUM_VAE_ENCODER_LAYERS"))
       },
       "mean_projection": {
        "activations": [ "linear" ],
        "input_dim": std.extVar("VAE_HIDDEN_DIM"),
        "hidden_dims": [ std.parseInt(std.extVar("VAE_LATENT_DIM")) ],
        "num_layers": 1
       },
      "log_variance_projection": {
        "activations": [ "linear" ],
        "input_dim": std.parseInt(std.extVar("VAE_HIDDEN_DIM")),
        "hidden_dims": [ std.parseInt(std.extVar("VAE_LATENT_DIM")) ],
        "num_layers": 1
      },

      // This will never get used but is included to recycle the VAE module.
      "decoder": {
        "activations": [ "tanh" ],
        "input_dim": std.parseInt(std.extVar("VAE_HIDDEN_DIM")),
        "hidden_dims": [ std.parseInt(std.extVar("M1_LATENT_DIM")) ],
        "num_layers": 1
       },
       "z_dropout": std.parseInt(std.extVar("Z_DROPOUT")) / 10.0
    },

    // The VAE the model uses to reconstruct z_1 from z_2.
    "reconstruction_vae": {
      "type": "logistic_normal",
      "apply_batchnorm": false,
      "encoder": {
        "activations": [ "softplus" for x in std.range(0, std.parseInt(std.extVar("RECONSTRUCTION_NUM_VAE_ENCODER_LAYERS")) - 1) ],

        // [z_2 ; y],
        "input_dim": std.parseInt(std.extVar("VAE_LATENT_DIM")) + std.parseInt(std.extVar("NUM_CLASSES")),
        // For simplicity and convenience, the hidden dim will be the same as the input.
        "hidden_dims": [ std.parseInt(std.extVar("VAE_LATENT_DIM")) for x in  std.range(0, std.parseInt(std.extVar("RECONSTRUCTION_NUM_VAE_ENCODER_LAYERS")) - 1) ],
        "num_layers": std.parseInt(std.extVar("RECONSTRUCTION_NUM_VAE_ENCODER_LAYERS"))
      },
      "mean_projection": {
        "activations": [ "linear" ],
        "input_dim": std.extVar("VAE_LATENT_DIM"),
        "hidden_dims": [ std.parseInt(std.extVar("M1_LATENT_DIM")) ],
        "num_layers": 1
      },
      "log_variance_projection": {
        "activations": [ "linear" ],
        "input_dim": std.parseInt(std.extVar("VAE_LATENT_DIM")),
        "hidden_dims": [ std.parseInt(std.extVar("M1_LATENT_DIM")) ],
        "num_layers": 1
      },

      // For the same reason the decoder of the VAE is not used, this is also never used.
      "decoder": {
        "activations": [ "tanh" ],
        "input_dim": std.parseInt(std.extVar("M1_LATENT_DIM")),
        "hidden_dims": [ std.parseInt(std.extVar("VAE_LATENT_DIM")) ],
        "num_layers": 1
      },
      "z_dropout": std.parseInt(std.extVar("Z_DROPOUT")) / 10.0,
    },
  },
  "iterator": {
    "batch_size": 32 * (std.parseInt(std.extVar("UNLABELED_DATA_FACTOR")) + 1),
    "type": "basic",
    "track_epoch": true
  },
  "trainer": {
    "cuda_device": CUDA_DEVICE,
    "num_epochs": 200,
    "optimizer": {
      "lr": std.parseInt(std.extVar("LEARNING_RATE")) / 10000.0,
      "type": "adam"
    },
    "patience": 15,
    "validation_metric": "+accuracy"
  }
}
