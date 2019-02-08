UNSUPERVISED_VAE = {
        "SEED": 42,
        "EMBEDDING_DIM": 50,
        "NUM_ENCODER_LAYERS": 1,
        "VAE_LATENT_DIM": 50,
        "VAE_HIDDEN_DIM": 100,
        "KL_ANNEALING": "sigmoid",
        "NUM_ENCODER_LAYERS": 1,
        "LEARNING_RATE": 1,
        "TRAIN_PATH": "s3://suching-dev/imdb/train.jsonl",
        "DEV_PATH": "s3://suching-dev/imdb/dev.jsonl",
        "REFERENCE_COUNTS": "s3://suching-dev/valid_npmi_reference/train.npz",
        "REFERENCE_VOCAB": "s3://suching-dev/valid_npmi_reference/train.vocab.json",
        "STOPWORDS_PATH": "s3://suching-dev/stopwords/snowball_stopwords.txt",
        "VOCAB_SIZE": 30000,
        "ADD_ELMO": 0,
        "USE_SPACY_TOKENIZER": 0,
        "THROTTLE": None,
        "NUM_GPU": 0
}


JOINT_VAE_LSTM = {
    "SEED": 42,
    "EMBEDDING_DIM": 50,
    "NUM_ENCODER_LAYERS": 1,
    "CLF_HIDDEN_DIM": 10,
    "AGGREGATIONS": "maxpool,meanpool",
    "ALPHA": 50,
    "KL_ANNEALING": "sigmoid",
    "VAE_LATENT_DIM": 50,
    "VAE_HIDDEN_DIM": 100,
    "LEARNING_RATE": 100,
    "CLASSIFIER": "lstm",
    "TRAIN_PATH": "s3://suching-dev/imdb/train.jsonl",
    "DEV_PATH": "s3://suching-dev/imdb/dev.jsonl",
    "REFERENCE_COUNTS": "s3://suching-dev/valid_npmi_reference/train.npz",
    "REFERENCE_VOCAB": "s3://suching-dev/valid_npmi_reference/train.vocab.json",
    "STOPWORDS_PATH": "s3://suching-dev/stopwords/snowball_stopwords.txt",
    "VOCAB_SIZE": 30000,
    "ADD_ELMO": 0,
    "USE_SPACY_TOKENIZER": 0,
    "THROTTLE": None,
    "NUM_GPU": 0
}


CLASSIFIER_LSTM = {
    "SEED": 42,
    "EMBEDDING_DIM": 50,
    "NUM_ENCODER_LAYERS": 1,
    "CLF_HIDDEN_DIM": 128,
    "AGGREGATIONS": "maxpool,meanpool",
    "LEARNING_RATE": 100,
    "CLASSIFIER": "lstm",
    "TRAIN_PATH": "s3://suching-dev/imdb/train.jsonl",
    "DEV_PATH": "s3://suching-dev/imdb/dev.jsonl",
    "REFERENCE_COUNTS": "s3://suching-dev/valid_npmi_reference/train.npz",
    "REFERENCE_VOCAB": "s3://suching-dev/valid_npmi_reference/train.vocab.json",
    "STOPWORDS_PATH": "s3://suching-dev/stopwords/snowball_stopwords.txt",
    "VOCAB_SIZE": 30000,
    "ADD_ELMO": 0,
    "ADD_VAE": 0,
    "USE_SPACY_TOKENIZER": 0,
    "THROTTLE": None,
    "NUM_GPU": 0
}

CLASSIFIER_CNN = {
    "SEED": 42,
    "EMBEDDING_DIM": 50,
    "NUM_FILTERS": 100,
    "CLF_HIDDEN_DIM": 128,
    "LEARNING_RATE": 100,
    "CLASSIFIER": "cnn",
    "TRAIN_PATH": "s3://suching-dev/imdb/train.jsonl",
    "DEV_PATH": "s3://suching-dev/imdb/dev.jsonl",
    "REFERENCE_COUNTS": "s3://suching-dev/valid_npmi_reference/train.npz",
    "REFERENCE_VOCAB": "s3://suching-dev/valid_npmi_reference/train.vocab.json",
    "STOPWORDS_PATH": "s3://suching-dev/stopwords/snowball_stopwords.txt",
    "VOCAB_SIZE": 30000,
    "ADD_ELMO": 0,
    "ADD_VAE": 0,
    "USE_SPACY_TOKENIZER": 0,
    "THROTTLE": None,
    "NUM_GPU": 0
}

CLASSIFIER_BOE = {
    "SEED": 42,
    "EMBEDDING_DIM": 50,
    "LEARNING_RATE": 100,
    "CLASSIFIER": "boe",
    "TRAIN_PATH": "s3://suching-dev/imdb/train.jsonl",
    "DEV_PATH": "s3://suching-dev/imdb/dev.jsonl",
    "REFERENCE_COUNTS": "s3://suching-dev/valid_npmi_reference/train.npz",
    "REFERENCE_VOCAB": "s3://suching-dev/valid_npmi_reference/train.vocab.json",
    "STOPWORDS_PATH": "s3://suching-dev/stopwords/snowball_stopwords.txt",
    "VOCAB_SIZE": 30000,
    "ADD_ELMO": 0,
    "ADD_VAE": 0,
    "USE_SPACY_TOKENIZER": 0,
    "THROTTLE": None,
    "NUM_GPU": 0
}

CLASSIFIER_LOGISTIC_REGRESSION = {
    "SEED": 42,
    "LEARNING_RATE": 100,
    "CLASSIFIER": "lr",
    "TRAIN_PATH": "s3://suching-dev/imdb/train.jsonl",
    "DEV_PATH": "s3://suching-dev/imdb/dev.jsonl",
    "REFERENCE_COUNTS": "s3://suching-dev/valid_npmi_reference/train.npz",
    "REFERENCE_VOCAB": "s3://suching-dev/valid_npmi_reference/train.vocab.json",
    "STOPWORDS_PATH": "s3://suching-dev/stopwords/snowball_stopwords.txt",
    "VOCAB_SIZE": 30000,
    "ADD_ELMO": 0,
    "ADD_VAE": 0,
    "USE_SPACY_TOKENIZER": 0,
    "THROTTLE": None,
    "NUM_GPU": 0
}

FIXED_ENVIRONMENTS = {
            'JOINT_VAE_LSTM': JOINT_VAE_LSTM,
            'UNSUPERVISED_VAE': UNSUPERVISED_VAE,
            "CLASSIFIER_LSTM": CLASSIFIER_LSTM,
            "CLASSIFIER_LOGISTIC_REGRESSION": CLASSIFIER_LOGISTIC_REGRESSION,
            "CLASSIFIER_CNN": CLASSIFIER_CNN
}