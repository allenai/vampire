from beaker.random_search import RandomSearch


CLASSIFIER_SEARCH = {
        "EMBEDDING_DIM": RandomSearch.random_choice(50, 128, 300, 512),
        "SEED": 42,
        "TRAIN_PATH": "s3://suching-dev/imdb/train.jsonl",
        "DEV_PATH": "s3://suching-dev/imdb/dev.jsonl",
        "REFERENCE_COUNTS": "s3://suching-dev/valid_npmi_reference/train.npz",
        "REFERENCE_VOCAB": "s3://suching-dev/valid_npmi_reference/train.vocab.json",
        "STOPWORDS_PATH": "s3://suching-dev/stopwords/snowball_stopwords.txt",
        "VOCAB_SIZE": 30000,
        "THROTTLE": None,
        "USE_SPACY_TOKENIZER": 1,
        "ADD_ELMO": 0,
        "ADD_VAE": 0,
        "LEARNING_RATE": RandomSearch.random_choice(1, 5, 10),
        "DROPOUT": RandomSearch.random_choice(0, 2, 5),
        "NUM_FILTERS": RandomSearch.random_choice(128, 156, 512),
        "MAX_FILTER_SIZE":  RandomSearch.random_choice(5, 7, 10),
        "NUM_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "AGGREGATIONS": RandomSearch.random_subset("final_state", "maxpool", "meanpool"),
        "CLF_HIDDEN_DIM": RandomSearch.random_choice(64, 128, 512, 1024, 2048),
        "CLASSIFIER": RandomSearch.random_choice("lstm", "boe", "lr", "cnn"),
        "NUM_GPU": 1
}

CNN_CLASSIFIER_SEARCH = {
        "EMBEDDING_DIM": RandomSearch.random_choice(50, 100, 300, 500),
        "SEED": 42,
        "TRAIN_PATH": "s3://suching-dev/imdb/train.jsonl",
        "DEV_PATH": "s3://suching-dev/imdb/dev.jsonl",
        "REFERENCE_COUNTS": "s3://suching-dev/valid_npmi_reference/train.npz",
        "REFERENCE_VOCAB": "s3://suching-dev/valid_npmi_reference/train.vocab.json",
        "STOPWORDS_PATH": "s3://suching-dev/stopwords/snowball_stopwords.txt",
        "VOCAB_SIZE": 30000,
        "THROTTLE": None,
        "USE_SPACY_TOKENIZER": 1,
        "ADD_ELMO": 0,
        "ADD_VAE": 0,
        "LEARNING_RATE": RandomSearch.random_choice(1, 5, 10),
        "NUM_FILTERS": RandomSearch.random_choice(128, 156, 512),
        "MAX_FILTER_SIZE":  RandomSearch.random_choice(5, 7, 10),
        "NUM_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "AGGREGATIONS": RandomSearch.random_subset("final_state", "maxpool", "meanpool"),
        "CLF_HIDDEN_DIM": RandomSearch.random_choice(64, 128, 512, 1024, 2048),
        "CLASSIFIER": "cnn",
        "NUM_GPU": 1
}

LSTM_CLASSIFIER_SEARCH = {
        "EMBEDDING_DIM": RandomSearch.random_choice(50, 100, 300, 500),
        "SEED": RandomSearch.random_integer(1, 2000000000),
        "TRAIN_PATH": "s3://suching-dev/imdb/train.jsonl",
        "DEV_PATH": "s3://suching-dev/imdb/dev.jsonl",
        "REFERENCE_COUNTS": "s3://suching-dev/valid_npmi_reference/train.npz",
        "REFERENCE_VOCAB": "s3://suching-dev/valid_npmi_reference/train.vocab.json",
        "STOPWORDS_PATH": "s3://suching-dev/stopwords/snowball_stopwords.txt",
        "VOCAB_SIZE": 30000,
        "THROTTLE": None,
        "USE_SPACY_TOKENIZER": 1,
        "ADD_ELMO": 0,
        "ADD_VAE": 0,
        "LEARNING_RATE": RandomSearch.random_choice(1, 5, 10),
        "NUM_FILTERS"
        "MAX_FILTER_SIZE":  RandomSearch.random_choice(5, 7, 10),
        "NUM_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "AGGREGATIONS": RandomSearch.random_subset("final_state", "maxpool", "meanpool"),
        "CLF_HIDDEN_DIM": RandomSearch.random_choice(64, 128, 512, 1024, 2048),
        "CLASSIFIER": "lstm",
        "NUM_GPU": 1
}



JOINT_VAE_SEARCH = {
        "EMBEDDING_DIM": RandomSearch.random_choice(50, 100, 300, 500),
        "SEED": 42,
        "TRAIN_PATH": "s3://suching-dev/imdb/train.jsonl",
        "DEV_PATH": "s3://suching-dev/imdb/dev.jsonl",
        "REFERENCE_COUNTS": "s3://suching-dev/valid_npmi_reference/train.npz",
        "REFERENCE_VOCAB": "s3://suching-dev/valid_npmi_reference/train.vocab.json",
        "STOPWORDS_PATH": "s3://suching-dev/stopwords/snowball_stopwords.txt",
        "VOCAB_SIZE": 30000,
        "THROTTLE": None,
        "USE_SPACY_TOKENIZER": 1,
        "ADD_ELMO": 0,
        "ADD_VAE": 0,
        "LEARNING_RATE": 1,
        "NUM_FILTERS": RandomSearch.random_choice(50, 100, 200),
        "NUM_CLF_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "AGGREGATIONS": RandomSearch.random_subset("final_state", "maxpool", "meanpool"),
        "CLF_HIDDEN_DIM": RandomSearch.random_choice(64, 128, 512, 1024, 2048),
        "LEARNING_RATE": RandomSearch.random_choice(1, 5, 10),
        "CLASSIFIER": RandomSearch.random_choice("lstm", "boe", "lr", "cnn"),
        "NUM_GPU": 1,
        "ALPHA": RandomSearch.random_integer(0, 50),
        "KL_ANNEALING": RandomSearch.random_choice('sigmoid', 'linear'),
        "VAE_HIDDEN_DIM":  RandomSearch.random_choice(64, 128, 512, 1024, 2048),
        "VAE_LATENT_DIM":  RandomSearch.random_choice(64, 128, 512, 1024, 2048),
        "NUM_VAE_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "NUM_GPU": 1
}


UNSUPERVISED_VAE_SEARCH = {
        "KL_ANNEALING": RandomSearch.random_choice('sigmoid', 'linear'),
        "VAE_HIDDEN_DIM":  RandomSearch.random_choice(64, 128, 512, 1024, 2048),
        "VAE_LATENT_DIM":  RandomSearch.random_choice(64, 128, 256, 512, 1024),
        "NUM_VAE_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "SEED" : 42,
        "LEARNING_RATE": RandomSearch.random_choice(1, 5, 10),
        "NUM_GPU": 1
}



SEARCH_ENVIRONMENTS = {
            'JOINT_VAE_SEARCH': JOINT_VAE_SEARCH,
            'UNSUPERVISED_VAE_SEARCH': UNSUPERVISED_VAE_SEARCH,
            "CLASSIFIER_SEARCH": CLASSIFIER_SEARCH,
            "CNN_CLASSIFIER_SEARCH": CNN_CLASSIFIER_SEARCH,
            "LSTM_CLASSIFIER_SEARCH": LSTM_CLASSIFIER_SEARCH
}