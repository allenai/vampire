import os
import argparse
import torch
import numpy as np
from allennlp.common.file_utils import cached_path
from allennlp.common.params import Params
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.archival import CONFIG_NAME, archive_model, load_archive
from allennlp.models.model import Model
from allennlp.predictors import Predictor
from allennlp.training.trainer import GradientDescentTrainer
from allennlp.common.util import prepare_environment
from vampire.data import VampireReader
from vampire.models import VAMPIRE
from vampire.predictors import VampirePredictor
from allennlp.modules.token_embedders import BagOfWordCountsTokenEmbedder
from vampire.modules.vae.logistic_normal import LogisticNormal
from vampire.models.vampire import KLAnneal, TrackLearningRate, ComputeTopics
from allennlp.modules.feedforward import FeedForward
from allennlp.training.optimizers import make_parameter_groups
from torch.optim import Adam
from allennlp.nn import Activation
from allennlp.data.dataloader import DataLoader
from allennlp.commands.train import TrainModel
def generate_config(seed,
                    z_dropout,
                    kld_clamp,
                    hidden_dim,
                    encoder_activation,
                    encoder_layers,
                    vocab_size,
                    batch_size,
                    cuda_device,
                    num_epochs,
                    patience,
                    learning_rate,
                    validation_metric,
                    train_path,
                    dev_path,
                    vocabulary_directory,
                    reference_counts,
                    reference_vocabulary,
                    background_data_path,
                    lazy,
                    sample,
                    min_sequence_length):
    PARAMS = {
        "numpy_seed": seed,
        "pytorch_seed": seed,
        "random_seed": seed,
        "dataset_reader": {
            "lazy": lazy,
            "sample": sample,
            "type": "vampire_reader",
            "min_sequence_length": min_sequence_length
        },
        "vocabulary": {
            "type": "from_files",
            "directory": vocabulary_directory
        },
        "train_data_path": train_path,
        "validation_data_path": dev_path,
        "model": {
            "type": "vampire",
            "bow_embedder": {
                "type": "bag_of_word_counts",
                "vocab_namespace": "vampire",
                "ignore_oov": True
            },
            "update_background_freq": False,
            "reference_counts": reference_counts,
            "reference_vocabulary": reference_vocabulary,
            "background_data_path": background_data_path,
            "vae": {
                "z_dropout": z_dropout,
                "kld_clamp": kld_clamp,

                "encoder": {
                    "activations": encoder_activation,
                    "hidden_dims": [hidden_dim] * encoder_layers,
                    "input_dim": vocab_size,
                    "num_layers": encoder_layers
                },
                "mean_projection": {
                    "activations": "linear",
                    "hidden_dims": [hidden_dim],
                    "input_dim": hidden_dim,
                    "num_layers": 1
                },
                "log_variance_projection": {
                    "activations": "linear",
                    "hidden_dims": [hidden_dim],
                    "input_dim": hidden_dim,
                    "num_layers": 1
                },
                "decoder": {
                    "activations": "linear",
                    "input_dim": hidden_dim,
                    "hidden_dims": [vocab_size],
                    "num_layers": 1 
                },
                "type": "logistic_normal"
            }
        },
        "data_loader": {
            "batch_sampler": {
                "type": "bucket",
                "batch_size": batch_size,
                "drop_last": False
            }
        },
        "trainer": {
            "epoch_callbacks": [{"type": "compute_topics"}, 
                                {"type": "kl_anneal", 
                                "kl_weight_annealing": "linear",
                                "linear_scaling": 1000}],
            "batch_callbacks": [{"type": "track_learning_rate"}],
            "cuda_device": cuda_device,
            "num_epochs": num_epochs,
            "patience": patience,
            "optimizer": {
                "lr": learning_rate,
                "type": "adam_str_lr"
            },
            "validation_metric": validation_metric,
            
        } 
    }
    return PARAMS
class VampireManager(object):

    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab

    @classmethod
    def from_pretrained(cls, pretrained_archive_path: str, cuda_device: int, for_prediction: bool) -> "VampireManager":
        if for_prediction:
            overrides = "{'model.reference_vocabulary': null}"
        else:
            overrides = None
        archive = load_archive(pretrained_archive_path, cuda_device=cuda_device, overrides=overrides)
        if for_prediction:
            model = Predictor.from_archive(archive, 'vampire')
            vocab = None
        else:
            model = Model.from_archive(archive, 'vampire')
            vocab = model._vocab
        return cls(model, vocab)

    @classmethod
    def from_params(cls, data_directory, kld_clamp, hidden_dim, vocab_size, ignore_npmi=False):
        vocab = Vocabulary.from_files(os.path.join(data_directory, 'vocabulary'))
        if not ignore_npmi:
            reference_counts = os.path.join(data_directory, "reference", "ref.npz")
            reference_vocabulary =os.path.join(data_directory, "reference", "ref.vocab.json")
        else:
            reference_counts = None
            reference_vocabulary = None
        background_data_path = os.path.join(data_directory, "vampire.bgfreq")
        bow_embedder = BagOfWordCountsTokenEmbedder(vocab, "vampire", ignore_oov=True)
        relu = Activation.by_name('relu')()
        linear = Activation.by_name('linear')()
        encoder = FeedForward(activations=relu,
                              input_dim=vocab.get_vocab_size('vampire'),
                              hidden_dims=[hidden_dim] * 2,
                              num_layers=2)
        mean_projection = FeedForward(activations=linear,
                              input_dim=hidden_dim,
                              hidden_dims=[hidden_dim],
                              num_layers=1)
        log_variance_projection = FeedForward(activations=linear,
                              input_dim=hidden_dim,
                              hidden_dims=[hidden_dim],
                              num_layers=1)
        decoder = FeedForward(activations=linear,
                              input_dim=hidden_dim,
                              hidden_dims=[vocab.get_vocab_size('vampire')],
                              num_layers=1)
        vae = LogisticNormal(vocab,
                             encoder,
                             mean_projection,
                             log_variance_projection,
                             decoder,
                             kld_clamp,
                             z_dropout=0.49)
        model = VAMPIRE(vocab=vocab,
                        reference_counts=reference_counts,
                        reference_vocabulary=reference_vocabulary,
                        background_data_path=background_data_path,
                        bow_embedder=bow_embedder,
                        vae=vae,
                        update_background_freq=False)
        return cls(model, vocab)
    
    def read_data(self,
                  train_path: str,
                  dev_path: str,
                  lazy: bool,
                  sample: int,
                  min_sequence_length: int):
        self.reader = VampireReader(lazy=lazy,
                                    sample=sample,
                                    min_sequence_length=min_sequence_length)
        train_dataset = self.reader.read(cached_path(train_path))
        validation_dataset = self.reader.read(cached_path(dev_path))
        return train_dataset, validation_dataset

    def fit(self,
            data_dir: str,
            serialization_dir: str,
            lazy: bool = False,
            sample: int = None,
            min_sequence_length:int=3,
            batch_size: int=512,
            kl_weight_annealing: str="linear",
            linear_scaling: int = 1000,
            cuda_device: int=-1,
            learning_rate: float=1e-3,
            num_epochs: int=50,
            patience: int=5,
            validation_metric: str='+npmi',
            seed: int = 0):
        train_path = os.path.join(data_dir, "train.npz")
        dev_path = os.path.join(data_dir, "dev.npz")
        reference_vocabulary = os.path.join(data_dir, "reference", "ref.vocab.json")
        reference_counts = os.path.join(data_dir, "reference","ref.npz")
        background_data_path = os.path.join(data_dir, "vampire.bgfreq")
        prepare_environment(Params({"pytorch_seed": seed, "numpy_seed": seed, "random_seed": seed}))
        vocabulary_path = os.path.join(serialization_dir, "vocabulary")
        os.mkdir(serialization_dir)
        config = generate_config(seed,
                    self.model.vae._z_dropout.p,
                    self.model.vae._kld_clamp,
                    self.model.vae.encoder.get_output_dim(),
                    "relu",
                    len(self.model.vae.encoder._linear_layers),
                    self.vocab.get_vocab_size('vampire'),
                    batch_size,
                    cuda_device,
                    num_epochs,
                    patience,
                    learning_rate,
                    validation_metric,
                    vocabulary_path,
                    train_path,
                    dev_path,
                    reference_counts,
                    reference_vocabulary,
                    background_data_path,
                    lazy,
                    sample,
                    min_sequence_length)
        Params(config).to_file(os.path.join(serialization_dir, CONFIG_NAME))
        self.vocab.save_to_files(vocabulary_path)
        os.environ['SEED'] = str(seed)
        train_dataset, validation_dataset = self.read_data(train_path, dev_path, lazy, sample, min_sequence_length)
        batch_sampler = BucketBatchSampler(train_dataset, batch_size=batch_size)
        data_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
        train_dataset.index_with(self.vocab)
        validation_dataset.index_with(self.vocab)
        validation_batch_sampler = BucketBatchSampler(validation_dataset, batch_size=batch_size)
        validation_data_loader = DataLoader(validation_dataset, batch_sampler=validation_batch_sampler)
        optimizer = Adam(lr=learning_rate, params=make_parameter_groups(list(self.model.named_parameters()), None))
        trainer = GradientDescentTrainer(model=self.model,
                                        serialization_dir=serialization_dir,
                                        validation_data_loader=validation_data_loader,
                                        data_loader=data_loader,
                                        epoch_callbacks=[KLAnneal(kl_weight_annealing=kl_weight_annealing,
                                                                    linear_scaling=linear_scaling),
                                                        ComputeTopics()],
                                        batch_callbacks=[TrackLearningRate()],
                                        cuda_device=cuda_device,
                                        num_epochs=num_epochs,
                                        patience=patience,
                                        optimizer=optimizer,
                                        validation_metric=validation_metric)
        train_loop = TrainModel(serialization_dir, self.model, trainer)
        metrics = train_loop.run()
        try:
            metrics = train_loop.run()
        except KeyboardInterrupt:
            # if we have completed an epoch, try to create a model archive.
            print(
                "Training interrupted by the user. Attempting to create "
                "a model archive using the current best epoch weights."
            )
            archive_model(serialization_dir)
            raise
        train_loop.finish(metrics)
        
        archive_model(serialization_dir) 
        return

    def predict(self, input_):
        return self.model.predict_json(input_)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden-dim', type=int, default=81)
    parser.add_argument('--kld-clamp', type=int, default=10000)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--seed', type=int, default=np.random.randint(0,10000000))
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--vocab-size', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)

    args = parser.parse_args()
    
    manager = VampireManager.from_params(args.data_dir,
                                         args.kld_clamp,
                                         args.hidden_dim,
                                         args.vocab_size,
                                         ignore_npmi=False)
    manager.fit(args.data_dir,
                "test/",
                seed=args.seed)
