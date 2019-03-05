// Number of GPUs to use. Setting this to zero will use the CPU.
local NUM_GPU = 0;

// Paths to data.
local TRAIN_PATH = "s3://suching-dev/imdb/train.jsonl";
local DEV_PATH = "s3://suching-dev/imdb/dev.jsonl";
local REFERENCE_COUNTS = "s3://suching-dev/valid_npmi_reference/train.npz";
local REFERENCE_VOCAB =  "s3://suching-dev/valid_npmi_reference/train.vocab.json";
local STOPWORDS_PATH =  "s3://suching-dev/stopwords/snowball_stopwords.txt";

// Vocabulary size
local VOCAB_SIZE = 10000;
// Throttle the training data to a random subset of this length.
local THROTTLE = 200;
// Use the SpaCy tokenizer when reading in the data. Set this to false if you'd like to debug faster.
local USE_SPACY_TOKENIZER = 0;

// Add ELMo embeddings to the input of the classifier.
local ADD_ELMO = 0;

// Add VAE embeddings to the input of the classifier.
local ADD_VAE = 0;

// learning rate of overall model.
local LEARNING_RATE = 0.001;

// type of classifier (choice between boe, cnn, lstm, and lr)
local CLASSIFIER = "boe";

// input embedding dimension to BOE
local EMBEDDING_DIM = 300;



local CUDA_DEVICE =
  if NUM_GPU == 0 then
    -1
  else if NUM_GPU > 1 then
    std.range(0, NUM_GPU - 1)
  else if NUM_GPU == 1 then
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

local VAE_FIELDS = {
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
                "representation": "encoder_output",
                "expand_dim": true,
                "model_archive": "s3://suching-dev/model.tar.gz",
                "background_frequency": "s3://suching-dev/vae.bgfreq.json",
                "dropout": 0.2
        }
    }
};

local VOCABULARY_WITH_VAE = {
  "vocabulary":{
              "type": "vocabulary_with_vae",
              "vae_vocab_file": "s3://suching-dev/vae.txt",
          }
};

// This reader will assume pre-tokenized input.
local BASE_JOINT_READER(ADD_ELMO, THROTTLE, UNLABELED_DATA_PATH, USE_SPACY_TOKENIZER, OVERSAMPLE) = {
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

local BASE_READER(ADD_ELMO, THROTTLE, UNLABELED_DATA_PATH, USE_SPACY_TOKENIZER, OVERSAMPLE) = {
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
  "unlabeled_data_path": UNLABELED_DATA_PATH

};


local BOE_CLF(EMBEDDING_DIM, ADD_ELMO, ADD_VAE) = {
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
            } + if ADD_VAE == 1 then VAE_FIELDS['vae_embedder'] else {}
              + if ADD_ELMO == 1 then ELMO_FIELDS['elmo_embedder'] else {}
         },
         
      
};

local CLF = BOE_CLF(EMBEDDING_DIM, ADD_ELMO, ADD_VAE);

{
   "numpy_seed": std.extVar("SEED"),
   "pytorch_seed": std.extVar("SEED"),
   "random_seed": std.extVar("SEED"),
   "dataset_reader": BASE_JOINT_READER(ADD_ELMO, THROTTLE, UNLABELED_DATA_PATH, USE_SPACY_TOKENIZER, true),
    "validation_dataset_reader": BASE_READER(ADD_ELMO, null, null, USE_SPACY_TOKENIZER, false),
   "datasets_for_vocab_creation": ["train"],
   "train_data_path": TRAIN_PATH,
   "validation_data_path": DEV_PATH,
   "model": {"type": "classifier"} + CLF,
    "iterator": {
      "batch_size": 128,
      "type": "basic"
   },
   "trainer": {
      "cuda_device": CUDA_DEVICE,
      "num_epochs": 200,
      "optimizer": {
         "lr": LEARNING_RATE,
         "type": "adam"
      },
      "patience": 20,
      "validation_metric": "+accuracy"
   }
} + if ADD_VAE == 1 then VOCABULARY_WITH_VAE else {}
