from beaker.random_search import RandomSearch
from beaker.datasets import DATASETS


###################################################################

from beaker.random_search import RandomSearch
from beaker.datasets import DATASETS


###################################################################


CLASSIFIER_SEARCH = {
        "EMBEDDING_DIM": RandomSearch.random_choice(50, 128, 300, 512),
        "SEED": RandomSearch.random_choice(1989894904, 2294922467, 2002866410, 1004506748, 4076792239),
        "ENCODER_ADDITIONAL_DIM": 0,
        "TRAIN_PATH": DATASETS['ag-news']['train'],
        "DEV_PATH": DATASETS['ag-news']['dev'],
        "REFERENCE_COUNTS": DATASETS['ag-news']['reference_counts'],
        "REFERENCE_VOCAB": DATASETS['ag-news']['reference_vocabulary'],
        "STOPWORDS_PATH": DATASETS['ag-news']['stopword_path'],
        "ELMO_OPTIONS_FILE": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
        "ELMO_WEIGHT_FILE": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
        "ELMO_DROPOUT": RandomSearch.random_choice(0, 2, 5),
        "ELMO_FINETUNE": False,
        "VAE_MODEL_ARCHIVE": "s3://suching-dev/best_npmi_vae/model.tar.gz",
        "VAE_BG_FREQ": "s3://suching-dev/best_npmi_vae/vae.bgfreq.json",
        "VAE_VOCAB": "s3://suching-dev/best_npmi_vae/vae.txt",
        "VAE_DROPOUT": RandomSearch.random_choice(0, 2, 5),
        "VOCAB_SIZE": 30000,
        "THROTTLE": 10000,
        "USE_SPACY_TOKENIZER": 1,
        "ADD_ELMO": 0,
        "ADD_VAE": 0,
        "ADD_BERT": 0,
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

UNSUPERVISED_VAE_SEARCH = {
        "KL_ANNEALING": 'sigmoid',
        "VAE_HIDDEN_DIM":  RandomSearch.random_choice(64, 128, 256, 512),
        "TRAIN_PATH": DATASETS['ag-news']['train'],
        "UNLABELED_DATA_PATH": None,
        "DEV_PATH": DATASETS['ag-news']['dev'],
        "REFERENCE_COUNTS": DATASETS['ag-news']['reference_counts'],
        "REFERENCE_VOCAB": DATASETS['ag-news']['reference_vocabulary'],
        "STOPWORDS_PATH": DATASETS['ag-news']['stopword_path'],
        "NUM_ENCODER_LAYERS": RandomSearch.random_choice(2, 3),
        "SEED": RandomSearch.random_choice(1989892904, 2294922667, 2002861410, 1004546748, 4076992239),
        "Z_DROPOUT": RandomSearch.random_choice(0, 2, 5),
        "LEARNING_RATE": RandomSearch.random_choice(1, 5, 10),
        "NUM_GPU": 1,
        "THROTTLE": None,
        "ADD_ELMO": 0,
        "USE_SPACY_TOKENIZER": 1,
        "VOCAB_SIZE": 30000,
        "VALIDATION_METRIC": RandomSearch.random_choice("+npmi")
}

FINE_TUNE_VAE = {
        "TRAIN_PATH": DATASETS['imdb']['train'],
        "UNLABELED_DATA_PATH": DATASETS['imdb']['unlabeled'],
        "DEV_PATH": DATASETS['imdb']['dev'],
        "REFERENCE_COUNTS": DATASETS['imdb']['reference_counts'],
        "REFERENCE_VOCAB": DATASETS['imdb']['reference_vocabulary'],
        "STOPWORDS_PATH": DATASETS['imdb']['stopword_path'],
        "SEED": RandomSearch.random_choice(1989892904, 2294922667, 2002861410, 1004546748, 4076992239),
        "LEARNING_RATE": RandomSearch.random_choice(1, 5, 10),
        "NUM_GPU": 0,
        "THROTTLE": 200,
        "ADD_ELMO": 0,
        "VAE_REQUIRES_GRAD": 1,
        "BATCH_SIZE": 32,
        "USE_SPACY_TOKENIZER": 1,
        "VOCAB_SIZE": 30000,
        "VAE_MODEL_ARCHIVE": "s3://suching-dev/best-npmi-vae-IMDB-final-big/model.tar.gz",
        "VAE_BG_FREQ": "s3://suching-dev/best-npmi-vae-IMDB-final-big/vae.bgfreq.json",
        "VAE_VOCAB": "s3://suching-dev/best-npmi-vae-IMDB-final-big/vae.txt",
}

UNSUPERVISED_VAE_SEARCH_1B = {
        "KL_ANNEALING": 'sigmoid',
        "VAE_HIDDEN_DIM":  RandomSearch.random_choice(64, 128, 256, 512),
        "TRAIN_PATH": DATASETS['1b']['train'],
        "UNLABELED_DATA_PATH": None,
        "DEV_PATH": DATASETS['1b']['test'],
        "REFERENCE_COUNTS": DATASETS['1b']['reference_counts'],
        "REFERENCE_VOCAB": DATASETS['1b']['reference_vocabulary'],
        "STOPWORDS_PATH": DATASETS['1b']['stopword_path'],
        "NUM_ENCODER_LAYERS": RandomSearch.random_choice(2, 3),
        "SEED": RandomSearch.random_choice(1989892904, 2294922667, 2002861410, 1004546748, 4076992239),
        "Z_DROPOUT": RandomSearch.random_choice(0, 2, 5),
        "LEARNING_RATE": RandomSearch.random_choice(1, 5, 10),
        "NUM_GPU": 1,
        "THROTTLE": None,
        "ADD_ELMO": 0,
        "USE_SPACY_TOKENIZER": 1,
        "VOCAB_SIZE": 30000,
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
        "ELMO_OPTIONS_FILE": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
        "ELMO_WEIGHT_FILE": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
        "ELMO_DROPOUT": RandomSearch.random_choice(0, 2, 5),
        "ELMO_FINETUNE": False,
        "VAE_MODEL_ARCHIVE": "s3://suching-dev/best-npmi-vae-IMDB-final-big/model.tar.gz",
        "VAE_BG_FREQ": "s3://suching-dev/best-npmi-vae-IMDB-final-big/vae.bgfreq.json",
        "VAE_VOCAB": "s3://suching-dev/best-npmi-vae-IMDB-final-big/vae.txt",
        "VAE_DROPOUT": RandomSearch.random_choice(0, 2, 5),
        "VOCAB_SIZE": 30000,
        "THROTTLE": 200,
        "USE_SPACY_TOKENIZER": 1,
        "ADD_ELMO": 0,
        "ADD_VAE": 1,
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

CLASSIFIER_WITH_BERT_SEARCH = {
        "EMBEDDING_DIM": RandomSearch.random_choice(50, 128, 300, 512),
        "SEED": RandomSearch.random_choice(1989894904, 2294922467, 2002866410, 1004506748, 4076792239),
        "ENCODER_ADDITIONAL_DIM": 768,
        "TRAIN_PATH": DATASETS['imdb']['train'],
        "DEV_PATH": DATASETS['imdb']['dev'],
        "REFERENCE_COUNTS": DATASETS['imdb']['reference_counts'],
        "REFERENCE_VOCAB": DATASETS['imdb']['reference_vocabulary'],
        "STOPWORDS_PATH": DATASETS['imdb']['stopword_path'],
        "ELMO_OPTIONS_FILE": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
        "ELMO_WEIGHT_FILE": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
        "ELMO_DROPOUT": RandomSearch.random_choice(0, 2, 5),
        "ELMO_FINETUNE": False,
        "VAE_MODEL_ARCHIVE": "s3://suching-dev/best_npmi_vae/model.tar.gz",
        "VAE_BG_FREQ": "s3://suching-dev/best_npmi_vae/vae.bgfreq.json",
        "VAE_VOCAB": "s3://suching-dev/best_npmi_vae/vae.txt",
        "VAE_DROPOUT": RandomSearch.random_choice(0, 2, 5),
        "VOCAB_SIZE": 30000,
        "THROTTLE": 200,
        "USE_SPACY_TOKENIZER": 1,
        "ADD_ELMO": 0,
        "ADD_VAE": 0,
        "ADD_BERT": 1,
        "BATCH_SIZE": 32,
        "LEARNING_RATE": RandomSearch.random_choice(1, 5, 10),
        "DROPOUT": RandomSearch.random_choice(0, 2, 5),
        "NUM_FILTERS": RandomSearch.random_choice(128, 156, 512),
        "MAX_FILTER_SIZE":  RandomSearch.random_choice(5, 7, 10),
        "NUM_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "AGGREGATIONS": RandomSearch.random_subset("final_state", "maxpool", "meanpool", "attention"),
        "CLF_HIDDEN_DIM": RandomSearch.random_choice(64, 128, 512),
        "CLASSIFIER": RandomSearch.random_choice("lstm", "boe", "cnn"),
        "NUM_GPU": 1
}

FINE_TUNE_ELMO = {
        "TRAIN_PATH": DATASETS['imdb']['train'],
        "DEV_PATH": DATASETS['imdb']['dev'],
        "SEED" : RandomSearch.random_choice(1989892904, 2294922667, 2002861410, 1004546748, 4076992239),
        "THROTTLE": 200,
        "USE_SPACY_TOKENIZER": 1,
        "BATCH_SIZE": 32,
        "NUM_GPU": 1,
}

FINE_TUNE_BERT = {
        "TRAIN_PATH": DATASETS['imdb']['train'],
        "DEV_PATH": DATASETS['imdb']['dev'],
        "SEED": RandomSearch.random_choice(1989892904, 2294922667, 2002861410, 1004546748, 4076992239),
        "THROTTLE": 200,
        "USE_SPACY_TOKENIZER": 1,
        "BATCH_SIZE": 32,
        "NUM_GPU": 1,
}


JOINT_VAE_SEARCH = {
        "KL_ANNEALING": RandomSearch.random_choice('sigmoid', 'linear'),

        # TODO(Tam) should there be annealing for the classification loss?
        # The best approach seems to be to "turn off" classificaton loss until
        # a certain point, just so the VAE has a chance to learn. Annealing may
        # cause severe overfitting by similuating a small learning rate.

        "VAE_HIDDEN_DIM":  RandomSearch.random_choice(64, 128, 256, 512, 1024, 2048),
        "VAE_LATENT_DIM":  RandomSearch.random_choice(64, 128, 256, 512, 1024),
        "TRAIN_PATH": DATASETS['imdb']['train'],
        "DEV_PATH": DATASETS['imdb']['dev'],
        "REFERENCE_COUNTS": DATASETS['imdb']['reference_counts'],
        "REFERENCE_VOCAB": DATASETS['imdb']['reference_vocabulary'],
        "STOPWORDS_PATH": DATASETS['imdb']['stopword_path'],
        "UNLABELED_DATA_PATH": DATASETS['imdb']['unlabeled'],
        "NUM_VAE_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "SEED" : 234,
        "Z_DROPOUT": RandomSearch.random_choice(0, 2, 5),
        "ADD_ELMO": 0,
        "EMBEDDING_DIM": RandomSearch.random_choice(50, 100, 300, 500),
        "THROTTLE": 2500,
        "USE_SPACY_TOKENIZER": 1,
        "LEARNING_RATE": RandomSearch.random_choice(1, 5, 10),
        "NUM_FILTERS": RandomSearch.random_choice(50, 100, 200),
        "MAX_FILTER_SIZE":  RandomSearch.random_choice(5, 7, 10),
        "NUM_CLF_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "AGGREGATIONS": RandomSearch.random_subset("final_state", "maxpool", "meanpool", "attention"),
        "CLF_HIDDEN_DIM": RandomSearch.random_choice(64, 128, 512, 1024, 2048),
        "DROPOUT": RandomSearch.random_choice(0, 2, 5),
        "CLASSIFIER": RandomSearch.random_choice("lstm", "boe", "lr", "cnn"),
        "NUM_GPU": 1,

        # Higher alpha tends to help in extreme throttlings.
        "ALPHA": RandomSearch.random_integer(1, 200),

        # Additons for accommodating resampling of unlabeled data.
        # In low data-regimes, batch size also has a noticeable impact.
        "BATCH_SIZE": RandomSearch.random_integer(20, 200),

        # In the 200 throttling, we can expect a vocab size of roughly 10K.
        # Change this for less drastic throttlings.
        "VOCAB_SIZE": 10000,

        # Determiens the proportion of unlabeled data. I.e. a value of 2
        # indicates a 2:1 ratio of unlabeled to labeled data.
        "UNLABELED_DATA_FACTOR": RandomSearch.random_choice(1, 2, 3, 4)
}



SEARCH_ENVIRONMENTS = {
            'JOINT_VAE_SEARCH': JOINT_VAE_SEARCH,
            'UNSUPERVISED_VAE_SEARCH': UNSUPERVISED_VAE_SEARCH,
            "CLASSIFIER_SEARCH": CLASSIFIER_SEARCH,
            "CLASSIFIER_WITH_NPMI_VAE_SEARCH": CLASSIFIER_WITH_NPMI_VAE_SEARCH,
            "FINE_TUNE_ELMO": FINE_TUNE_ELMO,
            "FINE_TUNE_BERT": FINE_TUNE_BERT,
            "FINE_TUNE_VAE": FINE_TUNE_VAE,
            "CLASSIFIER_WITH_BERT_SEARCH": CLASSIFIER_WITH_BERT_SEARCH,
            "UNSUPERVISED_VAE_SEARCH_1B": UNSUPERVISED_VAE_SEARCH_1B
}






