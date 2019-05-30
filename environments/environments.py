from environments.random_search import RandomSearch
from environments.datasets import DATASETS

CLASSIFIER = {
        "LAZY_DATASET_READER": 0,
        "CUDA_DEVICE": 0,
        "EVALUATE_ON_TEST": 1,
        "NUM_EPOCHS": 50,
        "SEED": 158361,
        "SEQUENCE_LENGTH": 400,
        "TRAIN_PATH": "s3://suching-dev/final-datasets/imdb/train_pretokenized.jsonl",
        "DEV_PATH": "s3://suching-dev/final-datasets/imdb/dev_pretokenized.jsonl",
        "TEST_PATH": "s3://suching-dev/final-datasets/imdb/test_pretokenized.jsonl",
        "THROTTLE": 200,
        "USE_SPACY_TOKENIZER": 0,
        "FREEZE_EMBEDDINGS": ["VAMPIRE"],
        "EMBEDDINGS": ["VAMPIRE", "RANDOM"],
        "ENCODER": "AVERAGE",
        "EMBEDDING_DROPOUT": 0.0,
        "LEARNING_RATE": 0.001,
        "DROPOUT": 0.5,
        "VAMPIRE_DIRECTORY": "logs/vampire_search/run_82_2019-05-26_01-41-541qqc8z1n 52",
        "BATCH_SIZE": 32,
        "NUM_ENCODER_LAYERS": 1,
        "NUM_OUTPUT_LAYERS": 2, 
        "MAX_FILTER_SIZE": RandomSearch.random_integer(3, 6),
        "NUM_FILTERS": RandomSearch.random_integer(64, 512),
        "HIDDEN_SIZE": RandomSearch.random_integer(64, 512),
        "AGGREGATIONS": RandomSearch.random_subset("maxpool", "meanpool", "attention", "final_state"),
        "MAX_CHARACTER_FILTER_SIZE": RandomSearch.random_integer(3, 6),
        "NUM_CHARACTER_FILTERS": RandomSearch.random_integer(16, 64),
        "CHARACTER_HIDDEN_SIZE": RandomSearch.random_integer(16, 128),
        "CHARACTER_EMBEDDING_DIM": RandomSearch.random_integer(16, 64),
        "CHARACTER_ENCODER": RandomSearch.random_choice("LSTM", "CNN", "AVERAGE"),
        "NUM_CHARACTER_ENCODER_LAYERS": RandomSearch.random_choice(1, 2),
}


VAMPIRE = {
        "LAZY_DATASET_READER": 0,
        "KL_ANNEALING": "sigmoid",
        "SIGMOID_WEIGHT_1": 0.3023115865397876,
        "SIGMOID_WEIGHT_2": 6.322409018592667,
        "LINEAR_SCALING": 60.827223157950584,
        "VAE_HIDDEN_DIM":  331,
        "ADDITIONAL_UNLABELED_DATA_PATH": None,
        "TRAIN_PATH": "/home/suching/vampire/train+unlabeled_pretokenized.jsonl",
        "DEV_PATH": DATASETS['imdb']['dev'],
        "REFERENCE_COUNTS": DATASETS['imdb']['reference_counts'],
        "REFERENCE_VOCAB": DATASETS['imdb']['reference_vocabulary'],
        "STOPWORDS_PATH": DATASETS['imdb']['stopword_path'],
        "NUM_ENCODER_LAYERS": 3,
        "ENCODER_ACTIVATION": "softplus",
        "MEAN_PROJECTION_ACTIVATION": "tanh",
        "NUM_MEAN_PROJECTION_LAYERS": 2,
        "LOG_VAR_PROJECTION_ACTIVATION": "softplus",
        "NUM_LOG_VAR_PROJECTION_LAYERS": 2,
        "PROJECTION_HIDDEN_DIM":  331,
        "DECODER_HIDDEN_DIM":  331,
        "DECODER_ACTIVATION": "linear",
        "DECODER_NUM_LAYERS": 1,
        "SEED": RandomSearch.random_integer(0, 100000),
        "Z_DROPOUT": 0.02,
        "LEARNING_RATE": 0.0019,
        "NUM_GPU": 0,
        "THROTTLE": None,
        "TRACK_NPMI": True,
        "ADD_ELMO": 0,
        "CUDA_DEVICE": 0,
        "USE_SPACY_TOKENIZER": 0,
        "UPDATE_BACKGROUND_FREQUENCY": 0,
        "VOCAB_SIZE": 30000,
        "APPLY_BATCHNORM": 1,
        "APPLY_BATCHNORM_1": 0,
        "SEQUENCE_LENGTH": 400,
        "BATCH_SIZE": 64,
        "VALIDATION_METRIC": "+npmi"
}



VAMPIRE_FAST = {
        "LAZY_DATASET_READER": 0,
        "KL_ANNEALING": "sigmoid",
        "SIGMOID_WEIGHT_1": 0.3023115865397876,
        "SIGMOID_WEIGHT_2": 6.322409018592667,
        "LINEAR_SCALING": 60.827223157950584,
        "VAE_HIDDEN_DIM":  331,
        "ADDITIONAL_UNLABELED_DATA_PATH": None,
        "TRAIN_PATH": "/home/suching/vampire/train_unlabeled.npz",
        "DEV_PATH": "/home/suching/vampire/dev.npz",
        "REFERENCE_COUNTS": DATASETS['imdb']['reference_counts'],
        "REFERENCE_VOCAB": DATASETS['imdb']['reference_vocabulary'],
        "VOCABULARY_DIRECTORY": "/home/suching/vampire/vocab/",
        "STOPWORDS_PATH": DATASETS['imdb']['stopword_path'],
        "NUM_ENCODER_LAYERS": 3,
        "ENCODER_ACTIVATION": "softplus",
        "MEAN_PROJECTION_ACTIVATION": "tanh",
        "NUM_MEAN_PROJECTION_LAYERS": 2,
        "LOG_VAR_PROJECTION_ACTIVATION": "softplus",
        "NUM_LOG_VAR_PROJECTION_LAYERS": 2,
        "PROJECTION_HIDDEN_DIM":  331,
        "DECODER_HIDDEN_DIM":  331,
        "DECODER_ACTIVATION": "linear",
        "DECODER_NUM_LAYERS": 1,
        "SEED": RandomSearch.random_integer(0, 100000),
        "Z_DROPOUT": 0.02,
        "LEARNING_RATE": 0.0019,
        "NUM_GPU": 0,
        "THROTTLE": None,
        "TRACK_NPMI": True,
        "ADD_ELMO": 0,
        "CUDA_DEVICE": 0,
        "USE_SPACY_TOKENIZER": 0,
        "UPDATE_BACKGROUND_FREQUENCY": 0,
        "VOCAB_SIZE": 10000,
        "APPLY_BATCHNORM": 1,
        "APPLY_BATCHNORM_1": 0,
        "BATCH_SIZE": 64,
        "VALIDATION_METRIC": "+npmi"
}


ENVIRONMENTS = {
        'VAMPIRE': VAMPIRE,
        'VAMPIRE_FAST': VAMPIRE_FAST,
        "CLASSIFIER": CLASSIFIER,
}






