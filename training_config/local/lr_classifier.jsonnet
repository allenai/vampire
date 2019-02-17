// Number of GPUs to use. Setting this to zero will use the CPU.
local NUM_GPU = 0;

// Paths to data.
local TRAIN_PATH = "s3://suching-dev/final-datasets/imdb/train.jsonl";
local DEV_PATH = "s3://suching-dev/final-datasets/imdb/dev.jsonl";
local REFERENCE_COUNTS = "s3://suching-dev/final-datasets/valid_npmi_reference/train.npz";
local REFERENCE_VOCAB =  "s3://suching-dev/final-datasets/valid_npmi_reference/train.vocab.json";
local STOPWORDS_PATH =  "s3://suching-dev/stopwords/snowball_stopwords.txt";

// Vocabulary size
local VOCAB_SIZE = 30000;
// Throttle the training data to a random subset of this length.
local THROTTLE = 10000;
// Use the SpaCy tokenizer when reading in the data. Set this to false if you'd like to debug faster.
local USE_SPACY_TOKENIZER = 1;

// Add ELMo embeddings to the input of the classifier.
local ADD_ELMO = 0;

// Add VAE embeddings to the input of the classifier.
local ADD_VAE = 1;

// learning rate of overall model.
local LEARNING_RATE = 0.0005;
local DROPOUT = 0.0;
local BATCH_SIZE = 32;
// type of classifier
local CLASSIFIER = "lr";


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
                "expand_dim": false,
                "model_archive": "s3://suching-dev/best-npmi-vae-IMDB-final-big/model.tar.gz",
                "background_frequency": "s3://suching-dev/best-npmi-vae-IMDB-final-big/vae.bgfreq.json",
                "dropout": 0.5
        }
    }
};

local VOCABULARY_WITH_VAE = {
  "vocabulary":{
              "type": "vocabulary_with_vae",
              "vae_vocab_file": "s3://suching-dev/best-npmi-vae-IMDB-final-big/vae.txt",
          }
};

local BASE_READER(ADD_ELMO, ADD_VAE, THROTTLE, USE_SPACY_TOKENIZER) = {
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
  } + if ADD_VAE == 1 then VAE_FIELDS['vae_indexer'] else {}
    + if ADD_ELMO == 1 then ELMO_FIELDS['elmo_indexer'] else {},
  "sequence_length": 400,
  "sample": THROTTLE,
};


local LR_CLF(ADD_VAE) = {
        "input_embedder": {
            "token_embedders": {
               "tokens": {
                  "type": "bag_of_word_counts",
                  "ignore_oov": "true",
                  "vocab_namespace": "classifier"
               }
            } + if ADD_VAE == 1 then VAE_FIELDS['vae_embedder'] else {}
         },
         "dropout": DROPOUT
};

local CLF = LR_CLF(ADD_VAE);
        

{
   "numpy_seed": std.extVar("SEED"),
   "pytorch_seed": std.extVar("SEED"),
   "random_seed": std.extVar("SEED"),
   "dataset_reader": BASE_READER(ADD_ELMO, ADD_VAE, THROTTLE, USE_SPACY_TOKENIZER),
    "validation_dataset_reader": BASE_READER(ADD_ELMO, ADD_VAE, null, USE_SPACY_TOKENIZER),
   "datasets_for_vocab_creation": ["train"],
   "train_data_path": TRAIN_PATH,
   "validation_data_path": DEV_PATH,
   "model": {"type": "classifier"} + CLF,
    "iterator": {
      "batch_size": BATCH_SIZE,
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
