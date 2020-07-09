local CUDA_DEVICE = std.parseInt(std.extVar("CUDA_DEVICE"));
local USE_LR_SCHEDULER = std.parseInt(std.extVar("USE_LR_SCHEDULER")) == 1;

local BASE_READER(LAZY, SAMPLE, MIN_SEQUENCE_LENGTH) = {
  "lazy": LAZY == 1,
  "sample": SAMPLE,
  "type": "vampire_wordvec_reader",
  "min_sequence_length": MIN_SEQUENCE_LENGTH
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

local LR_SCHEDULER = {
   "learning_rate_scheduler": {
        "type": "slanted_triangular",
        "num_epochs": std.parseInt(std.extVar("NUM_EPOCHS")),
        "num_steps_per_epoch": std.parseInt(std.extVar("DATASET_SIZE")) / std.parseInt(std.extVar("BATCH_SIZE")),
  },
};

{
   "numpy_seed": std.extVar("SEED"),
   "pytorch_seed": std.extVar("SEED"),
   "random_seed": std.extVar("SEED"),
   "dataset_reader": BASE_READER(std.parseInt(std.extVar("LAZY_DATASET_READER")), null, std.parseInt(std.extVar("MIN_SEQUENCE_LENGTH"))),
   "validation_dataset_reader": BASE_READER(std.parseInt(std.extVar("LAZY_DATASET_READER")), null,std.parseInt(std.extVar("MIN_SEQUENCE_LENGTH"))),
   "train_data_path": std.extVar("TRAIN_PATH"),
   "validation_data_path": std.extVar("DEV_PATH"),
//    "vocabulary": {
//       "type": "extended_vocabulary",
//       "directory_path": std.extVar("VOCABULARY_DIRECTORY")
//    },
   "model": {
      "type": "vampire",
      "bow_embedder": GLOVE_FIELDS(false)['glove_embedder']['tokens'],
      // "bow_embedder": {
      //    "type": "bag_of_word_counts",
      //    "vocab_namespace": "vampire",
      //    "ignore_oov": true
      // },
      "kl_weight_annealing": std.extVar("KL_ANNEALING"),
      "sigmoid_weight_1": std.extVar("SIGMOID_WEIGHT_1"),
      "sigmoid_weight_2": std.extVar("SIGMOID_WEIGHT_2"),
      "linear_scaling": std.extVar("LINEAR_SCALING"),
      "reference_counts": std.extVar("REFERENCE_COUNTS"),
      "reference_vocabulary": std.extVar("REFERENCE_VOCAB"),
      "update_background_freq": std.parseInt(std.extVar("UPDATE_BACKGROUND_FREQUENCY")) == 1,
      "track_npmi": std.parseInt(std.extVar("TRACK_NPMI")) == 1,
      "track_npmi_every_batch": std.parseInt(std.extVar("TRACK_NPMI_EVERY_BATCH")) == 1,
      "background_data_path": std.extVar("BACKGROUND_DATA_PATH"),
      "vae": {
         "z_dropout": std.extVar("Z_DROPOUT"),
         "kld_clamp": std.extVar("KLD_CLAMP"),
         "encoder": {
            "activations": std.makeArray(std.parseInt(std.extVar("NUM_ENCODER_LAYERS")), function(i) std.extVar("ENCODER_ACTIVATION")),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_ENCODER_LAYERS")), function(i) std.parseInt(std.extVar("VAE_HIDDEN_DIM"))),
            "input_dim": 50,
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
            "activations": "linear",
            "hidden_dims": [50],
            "input_dim": std.parseInt(std.extVar("VAE_HIDDEN_DIM")),
            "num_layers": 1
         },
         "type": "logistic_normal"
      }
      // "regularizer": [
      //   ["vae.encoder._linear_layers.*.weight", {"type": "l2", "alpha": std.extVar("REGULARIZATION_ALPHA")}],
      // ]
   },
    "iterator": {
      "batch_size": std.parseInt(std.extVar("BATCH_SIZE")),
      "track_epoch": false,
    //   "sorting_keys": [["tokens", "tokens_"]],
      "type": "basic"
   },
   "trainer": {
      "cuda_device": CUDA_DEVICE,
      "num_serialized_models_to_keep": 1,
      "num_epochs": std.parseInt(std.extVar("NUM_EPOCHS")),
      "patience": std.parseInt(std.extVar("PATIENCE")),
      "optimizer": {
         "lr": std.extVar("LEARNING_RATE"),
         "type": "adam"
      },
      "grad_norm": 5.0,
      "grad_clipping": 5.0,
      "validation_metric": std.extVar("VALIDATION_METRIC")
   } + if USE_LR_SCHEDULER then LR_SCHEDULER else {}
}
