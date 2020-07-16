PARAMS = {
        "numpy_seed": 42,
        "pytorch_seed": 42,
        "random_seed": 42,
    "model": {
        "bow_embedder": {
            "type": "bag_of_word_counts",
            "vocab_namespace": "vampire",
            "ignore_oov": True
        },
        "update_background_freq": False,
        "vae": {
            "z_dropout": 0.5,
            "kld_clamp": VAMPIRE_HPS['KLD_CLAMP'],
            "encoder": {
                "activations": VAMPIRE_HPS["ENCODER_ACTIVATION"],
                "hidden_dims": [VAMPIRE_HPS["VAE_HIDDEN_DIM"]] * VAMPIRE_HPS["NUM_ENCODER_LAYERS"],
                "input_dim": VAMPIRE_HPS["VOCAB_SIZE"] + 1,
                "num_layers": VAMPIRE_HPS["NUM_ENCODER_LAYERS"]
            },
            "mean_projection": {
                "activations": VAMPIRE_HPS["MEAN_PROJECTION_ACTIVATION"],
                "hidden_dims": [VAMPIRE_HPS["VAE_HIDDEN_DIM"]] * VAMPIRE_HPS["NUM_MEAN_PROJECTION_LAYERS"],
                "input_dim": VAMPIRE_HPS["VAE_HIDDEN_DIM"],
                "num_layers": VAMPIRE_HPS["NUM_MEAN_PROJECTION_LAYERS"]
            },
            "log_variance_projection": {
                "activations": VAMPIRE_HPS["LOG_VAR_PROJECTION_ACTIVATION"],
                "hidden_dims": VAMPIRE_HPS["NUM_LOG_VAR_PROJECTION_LAYERS"] * [VAMPIRE_HPS["VAE_HIDDEN_DIM"]],
                "input_dim": VAMPIRE_HPS["VAE_HIDDEN_DIM"],
                "num_layers": VAMPIRE_HPS["NUM_LOG_VAR_PROJECTION_LAYERS"]
            },
            "decoder": {
                "activations": "linear",
                "hidden_dims": [VAMPIRE_HPS["VOCAB_SIZE"] + 1],
                "input_dim": VAMPIRE_HPS["VAE_HIDDEN_DIM"],
                "num_layers": 1
            },
            "type": "logistic_normal"
        }
    },
    "data_loader": {
            "batch_sampler": {
                "type": "basic",
                "sampler": "sequential",
                "batch_size": VAMPIRE_HPS["BATCH_SIZE"],
                "drop_last": False
            }
        },
    "trainer": {
        "epoch_callbacks": [{"type": "compute_topics"}, 
                            {"type": "kl_anneal", 
                            "kl_weight_annealing": VAMPIRE_HPS["KL_ANNEALING"],
                            "sigmoid_weight_1": VAMPIRE_HPS["SIGMOID_WEIGHT_1"],
                            "sigmoid_weight_2": VAMPIRE_HPS["SIGMOID_WEIGHT_2"],
                            "linear_scaling": VAMPIRE_HPS["LINEAR_SCALING"]}],
        "batch_callbacks": [{"type": "track_learning_rate"}],
        "cuda_device": VAMPIRE_HPS['CUDA_DEVICE'],
        "num_epochs": VAMPIRE_HPS["NUM_EPOCHS"],
        "patience": VAMPIRE_HPS["PATIENCE"],
        "optimizer": {
            "lr": VAMPIRE_HPS["LEARNING_RATE"],
            "type": "adam_str_lr"
        },
        "validation_metric": VAMPIRE_HPS["VALIDATION_METRIC"],
        
    } 
    }