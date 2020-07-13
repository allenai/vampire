local CUDA_DEVICE = std.parseInt(std.extVar("CUDA_DEVICE"));

local USE_LR_SCHEDULER = std.parseInt(std.extVar("USE_LR_SCHEDULER"));


local BASE_READER(LAZY, SAMPLE, MIN_SEQUENCE_LENGTH) = {
  "lazy": LAZY == 1,
  "sample": SAMPLE,
  "type": "vampire_wordvec_reader",
  "min_sequence_length": MIN_SEQUENCE_LENGTH
};


local LR_SCHEDULER =  {
        "type": "slanted_triangular_float_num_steps",
        "num_epochs": std.parseInt(std.extVar("NUM_EPOCHS")),
        "num_steps_per_epoch": std.parseInt(std.extVar("DATASET_SIZE")) / std.parseInt(std.extVar("BATCH_SIZE")),
    };
{
   "numpy_seed": std.extVar("SEED"),
   "pytorch_seed": std.extVar("SEED"),
   "random_seed": std.extVar("SEED"),
   "dataset_reader": BASE_READER(std.parseInt(std.extVar("LAZY_DATASET_READER")), null, std.parseInt(std.extVar("MIN_SEQUENCE_LENGTH"))),
   "validation_dataset_reader": BASE_READER(std.parseInt(std.extVar("LAZY_DATASET_READER")), null,std.parseInt(std.extVar("MIN_SEQUENCE_LENGTH"))),
   "train_data_path": std.extVar("TRAIN_PATH"),
   "validation_data_path": std.extVar("DEV_PATH"),
   "vocabulary": {
      "type": "from_files",
      "directory": std.extVar("VOCABULARY_DIRECTORY")
   },
   "model": {
      "type": "vampire",
      "bow_embedder":
        {
        "tokens": {
          "embedding_dim": 300,
          "trainable": false,
          "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.50d.txt.gz",
        }
      },
      "reference_counts": std.extVar("REFERENCE_COUNTS"),
      "reference_vocabulary": std.extVar("REFERENCE_VOCAB"),
      "update_background_freq": std.parseInt(std.extVar("UPDATE_BACKGROUND_FREQUENCY")) == 1,
      "background_data_path": std.extVar("BACKGROUND_DATA_PATH"),
      "vae": {
         "z_dropout": std.extVar("Z_DROPOUT"),
         "kld_clamp": std.extVar("KLD_CLAMP"),
         "encoder": {
            "activations": std.makeArray(std.parseInt(std.extVar("NUM_ENCODER_LAYERS")), function(i) std.extVar("ENCODER_ACTIVATION")),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_ENCODER_LAYERS")), function(i) std.parseInt(std.extVar("VAE_HIDDEN_DIM"))),
            "input_dim": std.parseInt(std.extVar("VOCAB_SIZE")) + 1,
            "num_layers": std.parseInt(std.extVar("NUM_ENCODER_LAYERS"))
         },
         "mean_projection": {
            "activations": std.extVar("MEAN_PROJECTION_ACTIVATION"),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_MEAN_PROJECTION_LAYERS")), function(i) std.parseInt(std.extVar("VAE_HIDDEN_DIM"))),
            "input_dim": std.parseInt(std.extVar("VAE_HIDDEN_DIM")),
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
            "hidden_dims": [std.parseInt(std.extVar("VOCAB_SIZE")) + 1],
            "input_dim": std.parseInt(std.extVar("VAE_HIDDEN_DIM")),
            "num_layers": 1
         },
         "type": "logistic_normal"
      }
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
      "epoch_callbacks": [{"type": "compute_topics"}, 
                          {"type": "kl_anneal", "kl_weight_annealing": std.extVar("KL_ANNEALING"),
      "sigmoid_weight_1": std.extVar("SIGMOID_WEIGHT_1"),
      "sigmoid_weight_2": std.extVar("SIGMOID_WEIGHT_2"),
      "linear_scaling": std.extVar("LINEAR_SCALING")},],
      "batch_callbacks": [{"type": "track_learning_rate"}],
      "cuda_device": CUDA_DEVICE,
      "num_epochs": std.parseInt(std.extVar("NUM_EPOCHS")),
      "patience": std.parseInt(std.extVar("PATIENCE")),
      "optimizer": {
         "lr": std.extVar("LEARNING_RATE"),
         "type": "adam_str_lr"
      },
      "validation_metric": std.extVar("VALIDATION_METRIC"),
    
   }  + if USE_LR_SCHEDULER == 1 then LR_SCHEDULER else {}
}
