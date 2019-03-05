from beaker.random_search import RandomSearch
from beaker.datasets import DATASETS


###################################################################

from beaker.random_search import RandomSearch
from beaker.datasets import DATASETS


###################################################################


# CLASSIFIER_SEARCH = {
#         "LAZY_DATASET_READER": 0,
#         "EMBEDDING_DIM": RandomSearch.random_choice(50, 128, 300, 512),
#         "SEED": RandomSearch.random_choice(1989894904, 2294922467, 2002866410, 1004506748, 4076792239),
#         "ENCODER_ADDITIONAL_DIM": 0,
#         "TRAIN_PATH": DATASETS['amazon']['train'],
#         "DEV_PATH": DATASETS['amazon']['dev'],
#         "REFERENCE_COUNTS": DATASETS['amazon']['reference_counts'],
#         "REFERENCE_VOCAB": DATASETS['amazon']['reference_vocabulary'],
#         "STOPWORDS_PATH": DATASETS['amazon']['stopword_path'],
#         "ELMO_ARCHIVE_PATH": DATASETS['amazon']['elmo']['frozen'],
#         "GLOVE_PATH": DATASETS['amazon']['glove'],
#         "BERT_WEIGHTS": DATASETS['amazon']['bert']['weights'],
#         "BERT_VOCAB": DATASETS['amazon']['bert']['vocab'],
#         "ELMO_DROPOUT": RandomSearch.random_choice(0, 2, 5),
#         "ELMO_FINETUNE": False,
#         "VAE_MODEL_ARCHIVE": DATASETS['amazon']['vae']['model_archive'],
#         "VAE_BG_FREQ": DATASETS['amazon']['vae']['bg_freq'],
#         "VAE_VOCAB": DATASETS['amazon']['vae']['vocab'],
#         "VAE_DROPOUT": RandomSearch.random_choice(0, 2, 5),
#         "VOCAB_SIZE": 30000,
#         "THROTTLE": 10000,
#         "USE_SPACY_TOKENIZER": 1,
#         "ADD_ELMO": 0,
#         "ADD_VAE": 0,
#         "ADD_BERT": 0,
#         "ADD_BASIC": 1,
#         "BATCH_SIZE": 32,
#         "LEARNING_RATE": RandomSearch.random_choice(1, 5, 10),
#         "DROPOUT": RandomSearch.random_choice(0, 2, 5),
#         "NUM_FILTERS": RandomSearch.random_choice(128, 156, 512),
#         "MAX_FILTER_SIZE":  RandomSearch.random_choice(5, 7, 10),
#         "NUM_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
#         "AGGREGATIONS": RandomSearch.random_subset("final_state", "maxpool", "meanpool", "attention"),
#         "CLF_HIDDEN_DIM": RandomSearch.random_choice(64, 128, 512),
#         "CLASSIFIER": "boe",
#         "NUM_GPU": 1
# }

UNSUPERVISED_VAE_SEARCH = {
        "LAZY_DATASET_READER": 0,
        "KL_ANNEALING": 'sigmoid',
        "VAE_HIDDEN_DIM":  64,
        "TRAIN_PATH": DATASETS['imdb']['train'],
        "UNLABELED_DATA_PATH": DATASETS['imdb']['unlabeled'],
        "DEV_PATH": DATASETS['imdb']['dev'],
        "REFERENCE_COUNTS": DATASETS['imdb']['reference_counts'],
        "REFERENCE_VOCAB": DATASETS['imdb']['reference_vocabulary'],
        "STOPWORDS_PATH": DATASETS['imdb']['stopword_path'],
        "NUM_ENCODER_LAYERS": 2,
        "SEED": RandomSearch.random_choice(1989892904, 2294922667, 2002861410, 1004546748, 4076992239),
        "Z_DROPOUT": 2,
        "LEARNING_RATE": 10,
        "NUM_GPU": 0,
        "THROTTLE": None,
        "ADD_ELMO": 0,
        "USE_SPACY_TOKENIZER": 0,
        "VOCAB_SIZE": 10000,
        "VALIDATION_METRIC": RandomSearch.random_choice("+npmi")
}


# CLASSIFIER_WITH_NPMI_VAE_SEARCH = {
#         "EMBEDDING_DIM": RandomSearch.random_choice(50, 128, 300, 512),
#         "SEED": RandomSearch.random_choice(1989894904, 2294922467, 2002866410, 1004506748, 4076792239),
#         "ENCODER_ADDITIONAL_DIM": 512,
#         "TRAIN_PATH": DATASETS['imdb']['train'],
#         "DEV_PATH": DATASETS['imdb']['dev'],
#         "REFERENCE_COUNTS": DATASETS['imdb']['reference_counts'],
#         "REFERENCE_VOCAB": DATASETS['imdb']['reference_vocabulary'],
#         "STOPWORDS_PATH": DATASETS['imdb']['stopword_path'],
#         "ELMO_ARCHIVE_PATH": DATASETS['imdb']['elmo']['frozen'],
#         "GLOVE_PATH": DATASETS['imdb']['glove'],
#         "BERT_WEIGHTS": DATASETS['imdb']['bert']['weights'],
#         "BERT_VOCAB": DATASETS['imdb']['bert']['vocab'],
#         "ELMO_DROPOUT": RandomSearch.random_choice(0, 2, 5),
#         "ELMO_FINETUNE": False,
#         "VAE_MODEL_ARCHIVE": DATASETS['imdb']['vae']['model_archive'],
#         "VAE_BG_FREQ": DATASETS['imdb']['vae']['bg_freq'],
#         "VAE_VOCAB": DATASETS['imdb']['vae']['vocab'],
#         "VAE_DROPOUT": RandomSearch.random_choice(0, 2, 5),
#         "VOCAB_SIZE": 30000,
#         "THROTTLE": 200,
#         "USE_SPACY_TOKENIZER": 1,
#         "ADD_ELMO": 0,
#         "ADD_VAE": 1,
#         "ADD_BASIC": 1,
#         "ADD_BERT": 0,
#         "VAE_FINE_TUNE": 0,
#         "BATCH_SIZE": 32,
#         "LEARNING_RATE": RandomSearch.random_choice(1, 5, 10),
#         "DROPOUT": RandomSearch.random_choice(0, 2, 5),
#         "NUM_FILTERS": RandomSearch.random_choice(128, 156, 512),
#         "MAX_FILTER_SIZE":  RandomSearch.random_choice(5, 7, 10),
#         "NUM_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
#         "AGGREGATIONS": RandomSearch.random_subset("final_state", "maxpool", "meanpool", "attention"),
#         "CLF_HIDDEN_DIM": RandomSearch.random_choice(64, 128, 512),
#         "CLASSIFIER": RandomSearch.random_choice("lstm", "boe", "lr", "cnn"),
#         "NUM_GPU": 1
# }

# DATASET_TO_RUN = 'imdb-local-tam-deep'
DATASET_TO_RUN = 'ag-news'
NUM_GPU = 1
ADD_VAE = 1
VAE_FINETUNE = 0
ADD_BASIC = 1
ADD_GLOVE = 0
ADD_ELMO = 0
ELMO_FINETUNE = 0
ADD_BERT = 0
BERT_FINETUNE = 0
THROTTLE = 10000
ENCODER_INPUT_DIM = 50 + 512
# [161806,  51308, 156868, 93408, 158361, 49573, 5845, 64892, 108064,  23688]
"""
BOE_SMOKE_DEEP = {
        "EMBEDDING_DIM": 50,
        "ENCODER_INPUT_DIM": 50,
        "TRAIN_PATH": DATASETS[DATASET_TO_RUN]['train'],
        "DEV_PATH": DATASETS[DATASET_TO_RUN]['dev'],
        # "TEST_PATH": DATASETS[DATASET_TO_RUN]['test'],
        "EVALUATE_ON_TEST": 0,
        "REFERENCE_COUNTS": DATASETS[DATASET_TO_RUN]['reference_counts'],
        "REFERENCE_VOCAB": DATASETS[DATASET_TO_RUN]['reference_vocabulary'],
        "STOPWORDS_PATH": DATASETS[DATASET_TO_RUN]['stopword_path'],
        # "ELMO_ARCHIVE_PATH": DATASETS[DATASET_TO_RUN]['elmo']['in-domain'],
        # "GLOVE_PATH": DATASETS[DATASET_TO_RUN]['glove'],
        # "BERT_WEIGHTS": DATASETS[DATASET_TO_RUN]['bert']['weights'],
        # "BERT_VOCAB": DATASETS[DATASET_TO_RUN]['bert']['vocab'],
        "ELMO_DROPOUT": 0,
        "VAE_MODEL_ARCHIVE": DATASETS[DATASET_TO_RUN]['vae']['model_archive'],
        "VAE_BG_FREQ": DATASETS[DATASET_TO_RUN]['vae']['bg_freq'],
        "VAE_VOCAB": DATASETS[DATASET_TO_RUN]['vae']['vocab'],
        "VAE_DROPOUT": 5,
        "VOCAB_SIZE": 30000,
        "THROTTLE": THROTTLE,
        "USE_SPACY_TOKENIZER": 1,
        "ADD_ELMO": ADD_ELMO,
        "ADD_VAE": 0,
        "ADD_BASIC": ADD_BASIC,
        "ADD_BERT": ADD_BERT,
        "ADD_GLOVE": ADD_GLOVE,
        "ELMO_FINETUNE": ELMO_FINETUNE,
        "BERT_FINETUNE": BERT_FINETUNE,
        "VAE_FINETUNE": 0,
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 40,
        "DROPOUT": 5,
        "NUM_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "CLASSIFIER": "boe",
        "NUM_GPU": 1,
        "L1": 1, "L2": 1, "L3": 1

}
"""


BOE_SMOKE = {
        "SEED": 93408,
        "EMBEDDING_DIM": 50,
        "ENCODER_INPUT_DIM": ENCODER_INPUT_DIM,
        "TRAIN_PATH": DATASETS[DATASET_TO_RUN]['train'],
        "DEV_PATH": DATASETS[DATASET_TO_RUN]['dev'],
        "TEST_PATH": DATASETS[DATASET_TO_RUN]['test'],
        "EVALUATE_ON_TEST": 1,
        "REFERENCE_COUNTS": DATASETS[DATASET_TO_RUN]['reference_counts'],
        "REFERENCE_VOCAB": DATASETS[DATASET_TO_RUN]['reference_vocabulary'],
        "STOPWORDS_PATH": DATASETS[DATASET_TO_RUN]['stopword_path'],
        # "ELMO_ARCHIVE_PATH": DATASETS[DATASET_TO_RUN]['elmo']['in-domain'],
        "GLOVE_PATH": DATASETS[DATASET_TO_RUN]['glove']['in-domain'],
        # "BERT_WEIGHTS": DATASETS[DATASET_TO_RUN]['bert']['weights'],
        # "BERT_VOCAB": DATASETS[DATASET_TO_RUN]['bert']['vocab'],
        "ELMO_DROPOUT": 0,
        "VAE_MODEL_ARCHIVE": DATASETS[DATASET_TO_RUN]['vae']['model_archive'],
        "VAE_BG_FREQ": DATASETS[DATASET_TO_RUN]['vae']['bg_freq'],
        "VAE_VOCAB": DATASETS[DATASET_TO_RUN]['vae']['vocab'],
        "VAE_DROPOUT": 0,
        "VOCAB_SIZE": 30000,
        "THROTTLE": THROTTLE,
        "USE_SPACY_TOKENIZER": 0,
        "ADD_ELMO": ADD_ELMO,
        "ADD_VAE": ADD_VAE,
        "ADD_BASIC": ADD_BASIC,
        "ADD_BERT": ADD_BERT,
        "ADD_GLOVE": ADD_GLOVE,
        "ELMO_FINETUNE": ELMO_FINETUNE,
        "BERT_FINETUNE": BERT_FINETUNE,
        "VAE_FINETUNE": 0,
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 40,
        "DROPOUT": 5,
        "NUM_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "CLASSIFIER": "boe",
        "NUM_GPU": NUM_GPU,
        "L1": 1, "L2": -20, "L3": 1
}

BOE_HANDTUNE = {
        "EMBEDDING_DIM": 50,
        "ENCODER_INPUT_DIM": ENCODER_INPUT_DIM,
        "TRAIN_PATH": DATASETS[DATASET_TO_RUN]['train'],
        "DEV_PATH": DATASETS[DATASET_TO_RUN]['dev'],
        # "TEST_PATH": DATASETS[DATASET_TO_RUN]['test'],
        "EVALUATE_ON_TEST": 0,
        "REFERENCE_COUNTS": DATASETS[DATASET_TO_RUN]['reference_counts'],
        "REFERENCE_VOCAB": DATASETS[DATASET_TO_RUN]['reference_vocabulary'],
        "STOPWORDS_PATH": DATASETS[DATASET_TO_RUN]['stopword_path'],
        # "ELMO_ARCHIVE_PATH": DATASETS[DATASET_TO_RUN]['elmo']['in-domain'],
        # "GLOVE_PATH": DATASETS[DATASET_TO_RUN]['glove'],
        # "BERT_WEIGHTS": DATASETS[DATASET_TO_RUN]['bert']['weights'],
        # "BERT_VOCAB": DATASETS[DATASET_TO_RUN]['bert']['vocab'],
        "ELMO_DROPOUT": 0,
        "VAE_MODEL_ARCHIVE": DATASETS[DATASET_TO_RUN]['vae']['model_archive'],
        "VAE_BG_FREQ": DATASETS[DATASET_TO_RUN]['vae']['bg_freq'],
        "VAE_VOCAB": DATASETS[DATASET_TO_RUN]['vae']['vocab'],
        "VAE_DROPOUT": 5,
        "VOCAB_SIZE": 30000,
        "THROTTLE": THROTTLE,
        "USE_SPACY_TOKENIZER": 1,
        "ADD_ELMO": ADD_ELMO,
        "ADD_VAE": ADD_VAE,
        "ADD_BASIC": ADD_BASIC,
        "ADD_BERT": ADD_BERT,
        "ADD_GLOVE": ADD_GLOVE,
        "ELMO_FINETUNE": ELMO_FINETUNE,
        "BERT_FINETUNE": BERT_FINETUNE,
        "VAE_FINETUNE": VAE_FINETUNE,
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 40,
        "DROPOUT": 5,
        "NUM_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "CLASSIFIER": "boe",
        "NUM_GPU": 1
}


BOE = {
        "EMBEDDING_DIM": 50,
        "ENCODER_INPUT_DIM": ENCODER_INPUT_DIM,
        "TRAIN_PATH": DATASETS[DATASET_TO_RUN]['train'],
        "DEV_PATH": DATASETS[DATASET_TO_RUN]['dev'],
        "TEST_PATH": DATASETS[DATASET_TO_RUN]['test'],
        "EVALUATE_ON_TEST": 0,
        "REFERENCE_COUNTS": DATASETS[DATASET_TO_RUN]['reference_counts'],
        "REFERENCE_VOCAB": DATASETS[DATASET_TO_RUN]['reference_vocabulary'],
        "STOPWORDS_PATH": DATASETS[DATASET_TO_RUN]['stopword_path'],
        # "ELMO_ARCHIVE_PATH": DATASETS[DATASET_TO_RUN]['elmo']['in-domain'],
        # "GLOVE_PATH": DATASETS[DATASET_TO_RUN]['glove'],
        # "BERT_WEIGHTS": DATASETS[DATASET_TO_RUN]['bert']['weights'],
        # "BERT_VOCAB": DATASETS[DATASET_TO_RUN]['bert']['vocab'],
        "ELMO_DROPOUT": 0,
        "VAE_MODEL_ARCHIVE": DATASETS[DATASET_TO_RUN]['vae']['model_archive'],
        "VAE_BG_FREQ": DATASETS[DATASET_TO_RUN]['vae']['bg_freq'],
        "VAE_VOCAB": DATASETS[DATASET_TO_RUN]['vae']['vocab'],
        "VAE_DROPOUT": 5,
        "VOCAB_SIZE": 30000,
        "THROTTLE": THROTTLE,
        "USE_SPACY_TOKENIZER": 1,
        "ADD_ELMO": ADD_ELMO,
        "ADD_VAE": ADD_VAE,
        "ADD_BASIC": ADD_BASIC,
        "ADD_BERT": ADD_BERT,
        "ADD_GLOVE": ADD_GLOVE,
        "ELMO_FINETUNE": ELMO_FINETUNE,
        "BERT_FINETUNE": BERT_FINETUNE,
        "VAE_FINETUNE": VAE_FINETUNE,
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 40,
        "DROPOUT": 5,
        "NUM_FILTERS": RandomSearch.random_choice(128, 156, 512),
        "MAX_FILTER_SIZE":  RandomSearch.random_choice(5, 7, 10),
        "NUM_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "AGGREGATIONS": RandomSearch.random_subset("final_state", "maxpool", "meanpool", "attention"),
        "CLF_HIDDEN_DIM": RandomSearch.random_choice(64, 128, 512),
        "CLASSIFIER": "boe",
        "NUM_GPU": 0
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



# NUM_GPU = 1
# ADD_VAE = 1
# VAE_FINETUNE = 0
# ADD_BASIC = 1
# ADD_GLOVE = 0
# ADD_ELMO = 0
# ELMO_FINETUNE = 0
# ADD_BERT = 0
# BERT_FINETUNE = 0
# THROTTLE = 200
# ENCODER_INPUT_DIM = 50 + 512

DATASET_TO_RUN = 'yahoo'
THROTTLE = 200
JOINT_STACKED_VAE_SEARCH_IMDB = {

        # Toggle for baseline vs. stacked training.
        "BASELINE_ONLY": 0,
        "EVALUATE_ON_TEST": 1,
        # VAE config.
        "KL_ANNEALING": None,
        "VAE_LATENT_DIM": RandomSearch.random_choice(32, 64, 128),
        "M1_LATENT_DIM": 512, # This is not a hyperparameter: it should exactly match the VAE being used.
        "VAE_HIDDEN_DIM": RandomSearch.random_choice(128, 256, 512),
        "NUM_VAE_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),

        # Reconstruction VAE config.
        # "RECONSTRUCTION_VAE_HIDDEN_DIM": vae_latent_dim,
        "RECONSTRUCTION_NUM_VAE_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),

        # Paths.
        "TRAIN_PATH": DATASETS[DATASET_TO_RUN]['train'],
        "DEV_PATH": DATASETS[DATASET_TO_RUN]['dev'],
        "TEST_PATH": DATASETS[DATASET_TO_RUN]['test'],
        "REFERENCE_COUNTS": DATASETS[DATASET_TO_RUN]['reference_counts'],
        "REFERENCE_VOCAB": DATASETS[DATASET_TO_RUN]['reference_vocabulary'],
        "STOPWORDS_PATH": DATASETS[DATASET_TO_RUN]['stopword_path'],
        "UNLABELED_DATA_PATH": DATASETS[DATASET_TO_RUN]['unlabeled'],

        # VAE archive.
        "VAE_MODEL_ARCHIVE": DATASETS[DATASET_TO_RUN]['vae']['model_archive'],
        "VAE_BG_FREQ": DATASETS[DATASET_TO_RUN]['vae']['bg_freq'],
        "VAE_VOCAB": DATASETS[DATASET_TO_RUN]['vae']['vocab'],

        # MISC.
        "VAE_FINE_TUNE": 0,
        "VAE_DROPOUT": 0,
        "Z_DROPOUT": RandomSearch.random_choice(0, 2, 5),
        "LEARNING_RATE": RandomSearch.random_choice(1, 5, 10, 15),
        "NUM_CLF_ENCODER_LAYERS": 3,
        "NUM_GPU": 1,


        # Classifier type.
        "CLASSIFIER": "boe",

        # Applies to (most) encoders.
        "DROPOUT": RandomSearch.random_choice(0, 2, 5),

        # Fixed for consistency. 
        "EMBEDDING_DIM": 50,
        "ENCODER_INPUT_DIM": 50,

        # Flags for adding embeddings.
        "ADD_BERT": 0,
        "ADD_ELMO": 0,
        "ADD_GLOVE": 0,
        "ADD_BASIC": 1,
        "ADD_VAE": 0,

        "USE_SPACY_TOKENIZER": 1,

        # CNN-specific.
        "NUM_FILTERS": 156,
        "MAX_FILTER_SIZE":  7,
        "CLF_HIDDEN_DIM": 64,

        # Higher alpha tends to help in extreme throttlings.
        "ALPHA": RandomSearch.random_choice(10, 50, 100, 200),

        # Throttle.
        "THROTTLE": THROTTLE,

        # Additons for accommodating resampling of unlabeled data.
        # In low data-regimes, batch size also has a noticeable impact.
        # "BATCH_SIZE": 32 * (unlabeled_data_factor + 1),

        # Determiens the proportion of unlabeled data. I.e. a value of 2
        # indicates a 2:1 ratio of unlabeled to labeled data.
        "UNLABELED_DATA_FACTOR": RandomSearch.random_choice(1, 2, 3),

        # This value should match the number of categories the classifier
        # will predict from.
        "NUM_CLASSES": 15,
}



SEARCH_ENVIRONMENTS = {
            'BOE': BOE,
            'BOE_SMOKE': BOE_SMOKE,
            # 'BOE_SMOKE_DEEP': BOE_SMOKE_DEEP,
            'BOE_HANDTUNE': BOE_HANDTUNE,
            'JOINT_VAE_SEARCH': JOINT_VAE_SEARCH,
            'JOINT_STACKED_VAE_SEARCH_IMDB': JOINT_STACKED_VAE_SEARCH_IMDB,
            'UNSUPERVISED_VAE_SEARCH': UNSUPERVISED_VAE_SEARCH,
        #     "CLASSIFIER_SEARCH": CLASSIFIER_SEARCH,
        #     "CLASSIFIER_WITH_NPMI_VAE_SEARCH": CLASSIFIER_WITH_NPMI_VAE_SEARCH
}






