from beaker.random_search import RandomSearch

CLASSIFIER_SEARCH = {
        "EMBEDDING_DIM": RandomSearch.random_choice(50, 100, 300, 500),
        "SEED": 42,
        "NUM_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "AGGREGATIONS": RandomSearch.random_subset("final_state", "maxpool", "meanpool"),
        "CLF_HIDDEN_DIM": RandomSearch.random_choice(64, 128, 512, 1024, 2048),
        "LEARNING_RATE": 1,
        "CLASSIFIER": RandomSearch.random_choice("lstm", "boe", "lr", "cnn"),
        "NUM_GPU": 1
}


JOINT_VAE_LSTM_SEARCH = {
        "EMBEDDING_DIM": RandomSearch.random_choice(50, 100, 300, 500),
        "ALPHA": RandomSearch.random_integer(0, 50),
        "KL_ANNEALING": RandomSearch.random_choice('sigmoid', 'linear'),
        "VAE_HIDDEN_DIM":  RandomSearch.random_choice(64, 128, 512, 1024, 2048),
        "SEED" : 42,
        "NUM_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "AGGREGATIONS": RandomSearch.random_subset("final_state", "maxpool", "meanpool"),
        "CLF_HIDDEN_DIM": RandomSearch.random_choice(64, 128, 512, 1024, 2048),
        "LEARNING_RATE": 1,
        "CLASSIFIER": "lstm",
        "NUM_GPU": 1
}


UNSUPERVISED_VAE_SEARCH = {
        "EMBEDDING_DIM": RandomSearch.random_choice(50, 100, 300, 500),
        "ALPHA": RandomSearch.random_integer(0, 50),
        "KL_ANNEALING": RandomSearch.random_choice('sigmoid', 'linear'),
        "VAE_HIDDEN_DIM":  RandomSearch.random_choice(64, 128, 512, 1024, 2048),
        "VAE_LATENT_DIM":  RandomSearch.random_choice(64, 128, 256, 512, 1024),
        "SEED" : 42,
        "LEARNING_RATE": 1,
        "NUM_GPU": 1
}



SEARCH_ENVIRONMENTS = {
            'JOINT_VAE_LSTM': JOINT_VAE_LSTM_SEARCH,
            'UNSUPERVISED_VAE': UNSUPERVISED_VAE_SEARCH,
            "CLASSIFIER_LSTM": CLASSIFIER_LSTM_SEARCH
}