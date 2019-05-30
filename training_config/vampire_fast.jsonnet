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

local BASE_READER(THROTTLE, ADDITIONAL_UNLABELED_DATA_PATH, USE_SPACY_TOKENIZER, SEQUENCE_LENGTH, LAZY) = {
  "lazy": LAZY == 1,
  "type": "vampire_search",
  "ignore_labels": true,
  "sample": THROTTLE
};

{
   "numpy_seed": std.extVar("SEED"),
   "pytorch_seed": std.extVar("SEED"),
   "random_seed": std.extVar("SEED"),
   "dataset_reader": BASE_READER(std.extVar("THROTTLE"),
                                 std.extVar("ADDITIONAL_UNLABELED_DATA_PATH"),
                                 std.parseInt(std.extVar("USE_SPACY_TOKENIZER")),
                                 std.parseInt(std.extVar("SEQUENCE_LENGTH")),
                                 std.parseInt(std.extVar("LAZY_DATASET_READER"))),
   "validation_dataset_reader": BASE_READER(null,
                                            null,
                                            std.parseInt(std.extVar("USE_SPACY_TOKENIZER")),
                                            std.parseInt(std.extVar("SEQUENCE_LENGTH")),
                                            std.parseInt(std.extVar("LAZY_DATASET_READER"))),
   "datasets_for_vocab_creation": [
      "train"
   ],
   "train_data_path": std.extVar("TRAIN_PATH"),
   "validation_data_path": std.extVar("DEV_PATH"),
   "vocabulary": {
      "directory_path": std.extVar("VOCABULARY_DIRECTORY"),
      "type": "extended_vocabulary"
   },
   "model": {
      "type": "vampire_fast",
      "apply_batchnorm": std.parseInt(std.extVar("APPLY_BATCHNORM")) == 1,
      "bow_embedder": {
         "type": "bag_of_word_counts",
         "vocab_namespace": "vae",
         "ignore_oov": true
      },
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
            "input_dim": std.parseInt(std.extVar("VOCAB_SIZE")) + 1,
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
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("DECODER_NUM_LAYERS")),
                                         function(i) std.parseInt(std.extVar("DECODER_HIDDEN_DIM"))) + [std.parseInt(std.extVar("VOCAB_SIZE")) + 1],
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
      "num_epochs": 50,
      "patience": 5,
      "optimizer": {
         "lr": std.extVar("LEARNING_RATE"),
         "type": "adam"
      },
      "validation_metric": std.extVar("VALIDATION_METRIC")
   }
}
