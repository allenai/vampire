local CUDA_DEVICE = std.parseInt(std.extVar("CUDA_DEVICE"));

local BASE_READER(THROTTLE, USE_SPACY_TOKENIZER, SEQUENCE_LENGTH, LAZY) = {
  "lazy": LAZY == 1,
  "type": "semisupervised_text_classification_json",
  "tokenizer": {
    "word_splitter": if USE_SPACY_TOKENIZER == 1 then "spacy" else "just_spaces",
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
  },
  "token_indexers": {
    "tokens": {
      "type": "single_id",
      "namespace": "vae",
      "lowercase_tokens": true
    }
  },
  "max_sequence_length": SEQUENCE_LENGTH,
  "ignore_labels": true,
  "sample": THROTTLE
};

{
   "numpy_seed": std.extVar("SEED"),
   "pytorch_seed": std.extVar("SEED"),
   "random_seed": std.extVar("SEED"),
   "dataset_reader": BASE_READER(std.extVar("THROTTLE"), std.parseInt(std.extVar("USE_SPACY_TOKENIZER")), std.parseInt(std.extVar("SEQUENCE_LENGTH")), std.parseInt(std.extVar("LAZY_DATASET_READER"))),
   "validation_dataset_reader": BASE_READER( null, std.parseInt(std.extVar("USE_SPACY_TOKENIZER")), std.parseInt(std.extVar("SEQUENCE_LENGTH")), std.parseInt(std.extVar("LAZY_DATASET_READER"))),
   "datasets_for_vocab_creation": [
      "train"
   ],
   "train_data_path": std.extVar("TRAIN_PATH"),
   "validation_data_path": std.extVar("DEV_PATH"),
   "vocabulary": {
      "max_vocab_size": {
         "tokens": std.parseInt(std.extVar("VOCAB_SIZE"))
      },
      "type": "extended_vocabulary"
   },
   "model": {
      "type": "vampire",
      "apply_batchnorm": true,
      "bow_embedder": {
         "type": "bag_of_word_counts",
         "vocab_namespace": "tokens",
         "ignore_oov": true
      },
      "kl_weight_annealing": std.extVar("KL_ANNEALING"),
      "reference_counts": std.extVar("REFERENCE_COUNTS"),
      "reference_vocabulary": std.extVar("REFERENCE_VOCAB"),
      "update_background_freq": false,
      "vae": {
         "z_dropout": std.parseInt(std.extVar("Z_DROPOUT")) / 10.0,
         "apply_batchnorm": false,
         "encoder": {
            "activations": [
               "softplus" for x in std.range(0, std.parseInt(std.extVar("NUM_ENCODER_LAYERS")) - 1)
            ],
            "hidden_dims": [
               std.parseInt(std.extVar("VAE_HIDDEN_DIM")) for x in  std.range(0, std.parseInt(std.extVar("NUM_ENCODER_LAYERS")) - 1)
            ],
            "input_dim": std.parseInt(std.extVar("VOCAB_SIZE")) + 2,
            "num_layers": std.parseInt(std.extVar("NUM_ENCODER_LAYERS"))
         },
         "mean_projection": {
            "activations": [
               "linear"
            ],
            "hidden_dims": [
               std.parseInt(std.extVar("VAE_HIDDEN_DIM"))
            ],
            "input_dim": std.extVar("VAE_HIDDEN_DIM"),
            "num_layers": 1
         },
        "log_variance_projection": {
            "activations": [
               "linear"
            ],
            "hidden_dims": [
               std.parseInt(std.extVar("VAE_HIDDEN_DIM"))
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
            "input_dim": std.parseInt(std.extVar("VAE_HIDDEN_DIM")),
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
      "num_epochs": 50,
      "optimizer": {
         "lr": std.extVar("LEARNING_RATE"),
         "type": "adam"
      },
      "patience": 5,
      "validation_metric": std.extVar("VALIDATION_METRIC")
   }
}
