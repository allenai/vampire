// Number of GPUs to use. Setting this to zero will use the CPU.
local NUM_GPU = 0;


// Paths to data.
local TRAIN_PATH = "s3://suching-dev/imdb/train.jsonl";
local DEV_PATH = "s3://suching-dev/imdb/dev.jsonl";
local REFERENCE_COUNTS = "s3://suching-dev/valid_npmi_reference/train.npz";
local REFERENCE_VOCAB =  "s3://suching-dev/valid_npmi_reference/train.vocab.json";
local STOPWORDS_PATH =  "s3://suching-dev/stopwords/snowball_stopwords.txt";

// Vocabulary size
local VOCAB_SIZE = 30000;
// Throttle the training data to a random subset of this length.
local THROTTLE = null;
// Use the SpaCy tokenizer when reading in the data. Set this to false if you'd like to debug faster.
local USE_SPACY_TOKENIZER = 0;

// VAE-specific hyperparameters //
// KL divergence annealing (choice between constant, linear, and sigmoid)
local KL_ANNEALING = "linear";
// Number of encoder layers in VAE
local NUM_VAE_ENCODER_LAYERS = 1;
// dimension of latent space in VAE
local VAE_LATENT_DIM = 128;
// hidden dimension of the VAE encoder.
local VAE_HIDDEN_DIM = 512;

// learning rate of overall model.
local LEARNING_RATE = 0.001;

// Add ELMo embeddings to the input of the classifier.
local ADD_ELMO = 0;

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

local BASE_READER(ADD_ELMO, THROTTLE, USE_SPACY_TOKENIZER) = {
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
  "token_indexers": {
    "tokens": {
      "type": "single_id",
      "namespace": "vae",
      "lowercase_tokens": true
    }
  } + if ADD_ELMO == 1 then ELMO_FIELDS['elmo_indexer'] else {},
  "sequence_length": 400,
  "sample": THROTTLE,
};


{
   "numpy_seed": std.extVar("SEED"),
   "pytorch_seed": std.extVar("SEED"),
   "random_seed": std.extVar("SEED"),
   "dataset_reader": BASE_READER(ADD_ELMO, THROTTLE, USE_SPACY_TOKENIZER),
    "validation_dataset_reader": BASE_READER(ADD_ELMO, null, USE_SPACY_TOKENIZER),
   "datasets_for_vocab_creation": [
      "train"
   ],
   "train_data_path": TRAIN_PATH,
   "validation_data_path": DEV_PATH,
   "vocabulary": {
      "max_vocab_size": {
         "vae": VOCAB_SIZE
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
      "kl_weight_annealing": KL_ANNEALING,
      "reference_counts": REFERENCE_COUNTS,
      "reference_vocabulary": REFERENCE_VOCAB,
      "type": "nvdm",
      "update_background_freq": false,
      "vae": {
         "apply_batchnorm": false,
         "encoder": {
            "activations": [
               "softplus" for x in std.range(0, NUM_VAE_ENCODER_LAYERS - 1)
            ],
            "hidden_dims": [
               VAE_HIDDEN_DIM for x in  std.range(0, NUM_VAE_ENCODER_LAYERS - 1)
            ],
            "input_dim": VOCAB_SIZE + 2,
            "num_layers": NUM_VAE_ENCODER_LAYERS
         },
         "mean_projection": {
            "activations": [
               "linear"
            ],
            "hidden_dims": [
               VAE_LATENT_DIM
            ],
            "input_dim": VAE_HIDDEN_DIM,
            "num_layers": 1
         },
        "log_variance_projection": {
            "activations": [
               "linear"
            ],
            "hidden_dims": [
               VAE_LATENT_DIM
            ],
            "input_dim": VAE_HIDDEN_DIM,
            "num_layers": 1
         },
         "decoder": {
            "activations": [
               "tanh"
            ],
            "hidden_dims": [
               VOCAB_SIZE + 2
            ],
            "input_dim": VAE_LATENT_DIM,
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
         "lr": LEARNING_RATE,
         "type": "adam"
      },
      "patience": 20,
      "validation_metric": "+npmi"
   }
}
