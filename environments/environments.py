from environments.random_search import RandomSearch
from environments.datasets import DATASETS

DATA_DIR = "s3://suching-dev/final-datasets/imdb/"

CLASSIFIER_SEARCH = {
        "LAZY_DATASET_READER": 0,
        "CUDA_DEVICE": 0,
        "EVALUATE_ON_TEST": 0,
        "NUM_EPOCHS": 50,
        "SEED": RandomSearch.random_integer(0, 100),
        "DATA_DIR": DATA_DIR,
        "THROTTLE": 200,
        "USE_SPACY_TOKENIZER": 1,
        "FREEZE_EMBEDDINGS": ["VAMPIRE"],
        "EMBEDDINGS": ["VAMPIRE", "RANDOM"],
        "ENCODER": "CNN",
        "LEARNING_RATE": RandomSearch.random_loguniform(1e-6, 1e-1),
        "DROPOUT": RandomSearch.random_integer(0, 5),
        "BATCH_SIZE": 32,
        "NUM_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "NUM_OUTPUT_LAYERS": RandomSearch.random_choice(1, 2, 3),
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


VAMPIRE_SEARCH = {
        "LAZY_DATASET_READER": 0,
        "KL_ANNEALING": 'sigmoid',
        "VAE_HIDDEN_DIM":  64,
        "TRAIN_PATH": DATASETS['imdb']['train'],
        "DEV_PATH": DATASETS['imdb']['dev'],
        "REFERENCE_COUNTS": DATASETS['imdb']['reference_counts'],
        "REFERENCE_VOCAB": DATASETS['imdb']['reference_vocabulary'],
        "STOPWORDS_PATH": DATASETS['imdb']['stopword_path'],
        "NUM_ENCODER_LAYERS": 2,
        "SEED": RandomSearch.random_integer(0, 100),
        "Z_DROPOUT": 2,
        "LEARNING_RATE": 0.001,
        "NUM_GPU": 0,
        "THROTTLE": None,
        "ADD_ELMO": 0,
        "USE_SPACY_TOKENIZER": 0,
        "VOCAB_SIZE": 10000,
        "SEQUENCE_LENGTH": 400,
        "BATCH_SIZE": 32,
        "VALIDATION_METRIC": "+npmi"
}



ENVIRONMENTS = {
        'VAMPIRE_SEARCH': VAMPIRE_SEARCH,
        "CLASSIFIER_SEARCH": CLASSIFIER_SEARCH
}






