{
    "LAZY_DATASET_READER": 1,
    "KL_ANNEALING": {
        "sampling strategy": "choice",
        "choices": ["sigmoid", "linear"]
    },
    "SIGMOID_WEIGHT_1": 0.25,
    "SIGMOID_WEIGHT_2": 15, 
    "LINEAR_SCALING": 1000,
    "VAE_HIDDEN_DIM": {
        "sampling strategy": "integer",
        "bounds": [128, 1024]   
    },
    "TRAIN_PATH": "/home/suching/vampire/examples/tweets_sample/train.npz",
    "DEV_PATH": "/home/suching/vampire/examples/tweets_sample/dev.npz",
    "BACKGROUND_DATA_PATH":  "/home/suching/vampire/examples/tweets_sample/vampire.bgfreq",
    "VOCABULARY_DIRECTORY": "/home/suching/vampire/examples/tweets_sample/vocabulary/",
    "ADDITIONAL_UNLABELED_DATA_PATH": null,
    "REFERENCE_COUNTS": "/home/suching/vampire/examples/tweets_sample/reference/ref.npz",
    "REFERENCE_VOCAB": "/home/suching/vampire/examples/tweets_sample/reference/ref.vocab.json",
    "STOPWORDS_PATH": "s3://suching-dev/stopwords/snowball_stopwords.txt",
    "TRACK_NPMI": true,
    "NUM_ENCODER_LAYERS": {
        "sampling strategy": "choice",
        "choices": [2, 3]
    },
    "ENCODER_ACTIVATION": {
        "sampling strategy": "choice",
        "choices": ["relu", "tanh", "softplus"]
    },
    "NUM_MEAN_PROJECTION_LAYERS": 1,
    "MEAN_PROJECTION_ACTIVATION": "linear",
    "NUM_LOG_VAR_PROJECTION_LAYERS": 1,
    "LOG_VAR_PROJECTION_ACTIVATION": "linear",
    "SEED": {
        "sampling strategy": "integer",
        "bounds": [0, 100000]
    },
    "Z_DROPOUT": {
        "sampling strategy": "uniform",
        "bounds": [0, 0.5]
    },
    "LEARNING_RATE": {
        "sampling strategy": "loguniform",
        "bounds": [1e-3, 4e-3]
    },
    "CUDA_DEVICE": 0,
    "THROTTLE": null,
    "ADD_ELMO": 0,
    "PATIENCE": 10,
    "NUM_EPOCHS": 10,
    "USE_SPACY_TOKENIZER": 0,
    "UPDATE_BACKGROUND_FREQUENCY": 0,
    "VOCAB_SIZE": 30000,
    "APPLY_BATCHNORM": 1,
    "SEQUENCE_LENGTH": 400,
    "BATCH_SIZE": 64,
    "KLD_CLAMP":  {
        "sampling strategy": "uniform",
        "bounds": [1, 100000]
    },
    "VALIDATION_METRIC": "+npmi"
}