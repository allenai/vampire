from vampire.data import VampireReader
from allennlp.common.file_utils import cached_path 
from vampire.predictors import VampirePredictor
from vampire.models import VAMPIRE
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer
from allennlp.data import DataLoader


class VampireManager:

    def __init__(self, **kwargs):
        self.vocabulary = Vocabulary.from_params(Params(kwargs['vocabulary']))
        self.model = VAMPIRE.from_params(vocab=self.vocabulary, params=Params(kwargs['model']))
        
    def read_data(self, train_path: str, dev_path: str, **kwargs):
        reader = VampireReader.from_params(Params(kwargs['dataset_reader']))
        train_dataset = reader.read(cached_path(train_path))
        validation_dataset = reader.read(cached_path(dev_path))
        return train_dataset, validation_dataset

    def fit(self, train_path: str, dev_path: str, serialization_dir: str, **kwargs):
        train_dataset, validation_dataset = self.read_data(train_path, dev_path, **kwargs)
        data_loader = DataLoader.from_params(dataset=train_dataset, params=Params(kwargs['data_loader']))
        train_dataset.index_with(self.vocabulary)
        validation_dataset.index_with(self.vocabulary)
        validation_data_loader = DataLoader.from_params(dataset=validation_dataset, params=Params(kwargs['data_loader']))
        trainer = Trainer.from_params(model=self.model,
                               serialization_dir=serialization_dir,
                               validation_data_loader=validation_data_loader,
                               data_loader = data_loader,
                               params=Params(kwargs['trainer']))
        trainer.train()
    
    def predict(self, input_):
        predictor = VampirePredictor(self.model)
        predictor.predict_json(input_)

if __name__ == '__main__':
    DATA_DIR = "/home/suching/dont-stop-pretraining/vampire/data/world"
    VAMPIRE_HPS = {
            "LAZY_DATASET_READER": 0,
            "KL_ANNEALING": "linear",
            "KLD_CLAMP": 1000,
            "SIGMOID_WEIGHT_1": 0.25,
            "SIGMOID_WEIGHT_2": 15,
            "LINEAR_SCALING": 1000,
            "VAE_HIDDEN_DIM": 81,
            "TRAIN_PATH": DATA_DIR + "/train.npz",
            "DEV_PATH": DATA_DIR + "/dev.npz",
            "REFERENCE_COUNTS": DATA_DIR + "/reference/ref.npz",
            "REFERENCE_VOCAB": DATA_DIR + "/reference/ref.vocab.json",
            "VOCABULARY_DIRECTORY": DATA_DIR + "/vocabulary/",
            "BACKGROUND_DATA_PATH": DATA_DIR + "/vampire.bgfreq",
            "NUM_ENCODER_LAYERS": 2,
            "ENCODER_ACTIVATION": "relu",
            "MEAN_PROJECTION_ACTIVATION": "linear",
            "NUM_MEAN_PROJECTION_LAYERS": 1,
            "LOG_VAR_PROJECTION_ACTIVATION": "linear",
            "NUM_LOG_VAR_PROJECTION_LAYERS": 1,
            "SEED": 1,
            "Z_DROPOUT": 0.49,
            "LEARNING_RATE": 1e-3,
            "TRACK_NPMI": True,
            "CUDA_DEVICE": -1,
            "UPDATE_BACKGROUND_FREQUENCY": 0,
            "VOCAB_SIZE": 10000,
            "BATCH_SIZE": 512,
            "MIN_SEQUENCE_LENGTH": 3,
            "NUM_EPOCHS": 50,
            "PATIENCE": 5,
            "VALIDATION_METRIC": "+npmi",
            "USE_LR_SCHEDULER": 0
    }

    hps = {
    "numpy_seed": 0,
    "pytorch_seed": 0,
    "random_seed": 0,
    "dataset_reader": {
    "lazy": VAMPIRE_HPS['LAZY_DATASET_READER'],
    "sample": None,
    "min_sequence_length": VAMPIRE_HPS['MIN_SEQUENCE_LENGTH']
    },
    "train_data_path": VAMPIRE_HPS["TRAIN_PATH"],
    "validation_data_path": VAMPIRE_HPS["DEV_PATH"],
    "vocabulary": {
        "type": "from_files",
        "directory": VAMPIRE_HPS["VOCABULARY_DIRECTORY"]
    },
    "model": {
        "bow_embedder": {
            "type": "bag_of_word_counts",
            "vocab_namespace": "vampire",
            "ignore_oov": True
        },
        "reference_counts": VAMPIRE_HPS["REFERENCE_COUNTS"],
        "reference_vocabulary": VAMPIRE_HPS["REFERENCE_VOCAB"],
        "update_background_freq": False,
        "background_data_path": VAMPIRE_HPS["BACKGROUND_DATA_PATH"],
        "vae": {
            "z_dropout": 0.5,
            "kld_clamp": 10000,
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
    manager = VampireManager(**hps)
    manager.fit(hps['train_data_path'], hps['validation_data_path'], "test/", **hps)
