from beaker.random_search import RandomSearch
from beaker.datasets import DATASETS


###################################################################

from beaker.random_search import RandomSearch
from beaker.datasets import DATASETS


###################################################################


CLASSIFIER_SEARCH = {
        "LAZY_DATASET_READER": 0,
        "EMBEDDING_DIM": RandomSearch.random_choice(50, 128, 300, 512),
        "SEED": RandomSearch.random_choice(1989894904, 2294922467, 2002866410, 1004506748, 4076792239),
        "ENCODER_ADDITIONAL_DIM": 0,
        "TRAIN_PATH": DATASETS['amazon']['train'],
        "DEV_PATH": DATASETS['amazon']['dev'],
        "REFERENCE_COUNTS": DATASETS['amazon']['reference_counts'],
        "REFERENCE_VOCAB": DATASETS['amazon']['reference_vocabulary'],
        "STOPWORDS_PATH": DATASETS['amazon']['stopword_path'],
        "ELMO_ARCHIVE_PATH": DATASETS['amazon']['elmo']['frozen'],
        "GLOVE_PATH": DATASETS['amazon']['glove'],
        "BERT_WEIGHTS": DATASETS['amazon']['bert']['weights'],
        "BERT_VOCAB": DATASETS['amazon']['bert']['vocab'],
        "ELMO_DROPOUT": RandomSearch.random_choice(0, 2, 5),
        "ELMO_FINETUNE": False,
        "VAE_MODEL_ARCHIVE": DATASETS['amazon']['vae']['model_archive'],
        "VAE_BG_FREQ": DATASETS['amazon']['vae']['bg_freq'],
        "VAE_VOCAB": DATASETS['amazon']['vae']['vocab'],
        "VAE_DROPOUT": RandomSearch.random_choice(0, 2, 5),
        "VOCAB_SIZE": 30000,
        "THROTTLE": 10000,
        "USE_SPACY_TOKENIZER": 1,
        "ADD_ELMO": 0,
        "ADD_VAE": 0,
        "ADD_BERT": 0,
        "ADD_BASIC": 1,
        "BATCH_SIZE": 32,
        "LEARNING_RATE": RandomSearch.random_choice(1, 5, 10),
        "DROPOUT": RandomSearch.random_choice(0, 2, 5),
        "NUM_FILTERS": RandomSearch.random_choice(128, 156, 512),
        "MAX_FILTER_SIZE":  RandomSearch.random_choice(5, 7, 10),
        "NUM_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "AGGREGATIONS": RandomSearch.random_subset("final_state", "maxpool", "meanpool", "attention"),
        "CLF_HIDDEN_DIM": RandomSearch.random_choice(64, 128, 512),
        "CLASSIFIER": "boe",
        "NUM_GPU": 1
}

UNSUPERVISED_VAE_SEARCH = {
        "LAZY_DATASET_READER": 0,
        "KL_ANNEALING": 'sigmoid',
        "VAE_HIDDEN_DIM":  RandomSearch.random_choice(64, 128, 256, 512),
        "TRAIN_PATH": DATASETS['imdb']['train'],
        "UNLABELED_DATA_PATH": DATASETS['imdb']['unlabeled'],
        "DEV_PATH": DATASETS['imdb']['dev'],
        "REFERENCE_COUNTS": DATASETS['imdb']['reference_counts'],
        "REFERENCE_VOCAB": DATASETS['imdb']['reference_vocabulary'],
        "STOPWORDS_PATH": DATASETS['imdb']['stopword_path'],
        "NUM_ENCODER_LAYERS": RandomSearch.random_choice(2, 3),
        "SEED": RandomSearch.random_choice(1989892904, 2294922667, 2002861410, 1004546748, 4076992239),
        "Z_DROPOUT": RandomSearch.random_choice(0, 2, 5),
        "LEARNING_RATE": RandomSearch.random_choice(1, 5, 10),
        "NUM_GPU": 1,
        "THROTTLE": None,
        "ADD_ELMO": 0,
        "USE_SPACY_TOKENIZER": 1,
        "VOCAB_SIZE": 9000,
        "VALIDATION_METRIC": RandomSearch.random_choice("+npmi")
}


CLASSIFIER_WITH_NPMI_VAE_SEARCH = {
        "EMBEDDING_DIM": RandomSearch.random_choice(50, 128, 300, 512),
        "SEED": RandomSearch.random_choice(1989894904, 2294922467, 2002866410, 1004506748, 4076792239),
        "ENCODER_ADDITIONAL_DIM": 512,
        "TRAIN_PATH": DATASETS['imdb']['train'],
        "DEV_PATH": DATASETS['imdb']['dev'],
        "REFERENCE_COUNTS": DATASETS['imdb']['reference_counts'],
        "REFERENCE_VOCAB": DATASETS['imdb']['reference_vocabulary'],
        "STOPWORDS_PATH": DATASETS['imdb']['stopword_path'],
        "ELMO_ARCHIVE_PATH": DATASETS['imdb']['elmo']['frozen'],
        "GLOVE_PATH": DATASETS['imdb']['glove'],
        "BERT_WEIGHTS": DATASETS['imdb']['bert']['weights'],
        "BERT_VOCAB": DATASETS['imdb']['bert']['vocab'],
        "ELMO_DROPOUT": RandomSearch.random_choice(0, 2, 5),
        "ELMO_FINETUNE": False,
        "VAE_MODEL_ARCHIVE": DATASETS['imdb']['vae']['model_archive'],
        "VAE_BG_FREQ": DATASETS['imdb']['vae']['bg_freq'],
        "VAE_VOCAB": DATASETS['imdb']['vae']['vocab'],
        "VAE_DROPOUT": RandomSearch.random_choice(0, 2, 5),
        "VOCAB_SIZE": 30000,
        "THROTTLE": 200,
        "USE_SPACY_TOKENIZER": 1,
        "ADD_ELMO": 0,
        "ADD_VAE": 1,
        "ADD_BASIC": 1,
        "ADD_BERT": 0,
        "VAE_FINE_TUNE": 0,
        "BATCH_SIZE": 32,
        "LEARNING_RATE": RandomSearch.random_choice(1, 5, 10),
        "DROPOUT": RandomSearch.random_choice(0, 2, 5),
        "NUM_FILTERS": RandomSearch.random_choice(128, 156, 512),
        "MAX_FILTER_SIZE":  RandomSearch.random_choice(5, 7, 10),
        "NUM_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "AGGREGATIONS": RandomSearch.random_subset("final_state", "maxpool", "meanpool", "attention"),
        "CLF_HIDDEN_DIM": RandomSearch.random_choice(64, 128, 512),
        "CLASSIFIER": RandomSearch.random_choice("lstm", "boe", "lr", "cnn"),
        "NUM_GPU": 1
}

BOE = {
        "EMBEDDING_DIM": 50,
        "SEED": RandomSearch.random_choice(2436, 144583, 78943, 150314, 135227, 59807, 180134, 91397, 45642, 118101),
        "ENCODER_INPUT_DIM": 50,
        "TRAIN_PATH": DATASETS['ag-news']['train'],
        "DEV_PATH": DATASETS['ag-news']['dev'],
        "REFERENCE_COUNTS": DATASETS['ag-news']['reference_counts'],
        "REFERENCE_VOCAB": DATASETS['ag-news']['reference_vocabulary'],
        "STOPWORDS_PATH": DATASETS['ag-news']['stopword_path'],
        "ELMO_ARCHIVE_PATH": DATASETS['ag-news']['elmo']['frozen'],
        "GLOVE_PATH": DATASETS['ag-news']['glove'],
        "BERT_WEIGHTS": DATASETS['ag-news']['bert']['weights'],
        "BERT_VOCAB": DATASETS['ag-news']['bert']['vocab'],
        "ELMO_DROPOUT": RandomSearch.random_choice(0, 2, 5),
        "ELMO_FINETUNE": False,
        "VAE_MODEL_ARCHIVE": DATASETS['ag-news']['vae']['model_archive'],
        "VAE_BG_FREQ": DATASETS['ag-news']['vae']['bg_freq'],
        "VAE_VOCAB": DATASETS['ag-news']['vae']['vocab'],
        "VAE_DROPOUT": RandomSearch.random_choice(1, 2, 5),
        "VOCAB_SIZE": 30000,
        "THROTTLE": 10000,
        "USE_SPACY_TOKENIZER": 1,
        "ADD_ELMO": 0,
        "ADD_VAE": 0,
        "ADD_BASIC": 1,
        "ADD_BERT": 0,
        "ADD_GLOVE": 0,
        "VAE_FINE_TUNE": 0,
        "BATCH_SIZE": 32,
        "LEARNING_RATE": RandomSearch.random_choice(1, 5, 10),
        "DROPOUT": RandomSearch.random_choice(1, 2, 5),
        "NUM_FILTERS": RandomSearch.random_choice(128, 156, 512),
        "MAX_FILTER_SIZE":  RandomSearch.random_choice(5, 7, 10),
        "NUM_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "AGGREGATIONS": RandomSearch.random_subset("final_state", "maxpool", "meanpool", "attention"),
        "CLF_HIDDEN_DIM": RandomSearch.random_choice(64, 128, 512),
        "CLASSIFIER": "boe",
        "NUM_GPU": 1
}

JOINT_VAE_SEARCH = {
        "KL_ANNEALING": RandomSearch.random_choice('sigmoid', 'linear', None),

        # TODO(Tam) should there be annealing for the classification loss?
        # The best approach seems to be to "turn off" classificaton loss until
        # a certain point, just so the VAE has a chance to learn. Annealing may
        # cause severe overfitting by similuating a small learning rate.

        "VAE_HIDDEN_DIM":  RandomSearch.random_choice(64, 128, 256, 512),
        "VAE_LATENT_DIM":  RandomSearch.random_choice(64, 128, 256, 512),
        "TRAIN_PATH": DATASETS['imdb']['train'],
        "DEV_PATH": DATASETS['imdb']['dev'],
        "REFERENCE_COUNTS": DATASETS['imdb']['reference_counts'],
        "REFERENCE_VOCAB": DATASETS['imdb']['reference_vocabulary'],
        "STOPWORDS_PATH": DATASETS['imdb']['stopword_path'],
        "UNLABELED_DATA_PATH": DATASETS['imdb']['unlabeled'],
        "NUM_VAE_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "Z_DROPOUT": RandomSearch.random_choice(0, 2, 5),
        "ADD_ELMO": 0,
        "EMBEDDING_DIM": RandomSearch.random_choice(50, 100, 300, 500),
        "THROTTLE": 200,
        "USE_SPACY_TOKENIZER": 1,
        "LEARNING_RATE": RandomSearch.random_choice(1, 5, 10),
        "NUM_FILTERS": RandomSearch.random_choice(50, 100, 200),
        "MAX_FILTER_SIZE":  RandomSearch.random_choice(5, 7, 10),
        "NUM_CLF_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "SEED": RandomSearch.random_choice(1989892904, 2294922667, 2002861410, 1004546748, 4076992239),
        "AGGREGATIONS": RandomSearch.random_subset("final_state", "maxpool", "meanpool", "attention"),
        "CLF_HIDDEN_DIM": RandomSearch.random_choice(64, 128, 512, 1024, 2048),
        "DROPOUT": RandomSearch.random_choice(0, 2, 5),
        "CLASSIFIER": RandomSearch.random_choice("lstm", "boe", "lr", "cnn"),
        "NUM_GPU": 1,

        # Higher alpha tends to help in extreme throttlings.
        "ALPHA": RandomSearch.random_choice(20, 40, 60, 80, 100),

        # Additons for accommodating resampling of unlabeled data.
        # In low data-regimes, batch size also has a noticeable impact.
        "BATCH_SIZE": RandomSearch.random_choice(32, 64, 128),

        # In the 200 throttling, we can expect a vocab size of roughly 10K.
        # Change this for less drastic throttlings.
        "VOCAB_SIZE": 10000,

        # Determiens the proportion of unlabeled data. I.e. a value of 2
        # indicates a 2:1 ratio of unlabeled to labeled data.
        "UNLABELED_DATA_FACTOR": RandomSearch.random_choice(1, 2, 3, 4)
}



SEARCH_ENVIRONMENTS = {
            'BOE': BOE,
            'JOINT_VAE_SEARCH': JOINT_VAE_SEARCH,
            'UNSUPERVISED_VAE_SEARCH': UNSUPERVISED_VAE_SEARCH,
            "CLASSIFIER_SEARCH": CLASSIFIER_SEARCH,
            "CLASSIFIER_WITH_NPMI_VAE_SEARCH": CLASSIFIER_WITH_NPMI_VAE_SEARCH
}






