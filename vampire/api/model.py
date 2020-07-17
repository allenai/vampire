import argparse
import logging
import os
import sys

import numpy as np
import torch
from typing import List
from allennlp.commands.train import TrainModel
from allennlp.common.file_utils import cached_path
from allennlp.common.params import Params
from allennlp.common.util import prepare_environment
from allennlp.data.dataloader import DataLoader
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.archival import CONFIG_NAME, archive_model, load_archive
from allennlp.models.model import Model
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.token_embedders import BagOfWordCountsTokenEmbedder
from allennlp.nn import Activation
from allennlp.predictors import Predictor
from allennlp.training.optimizers import make_parameter_groups
from allennlp.training.trainer import GradientDescentTrainer
from torch.optim import Adam
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from vampire.common.util import generate_config, save_sparse, write_to_json, write_list_to_file
from vampire.data import VampireReader
from vampire.models import VAMPIRE
from vampire.models.vampire import ComputeTopics, KLAnneal, TrackLearningRate
from vampire.modules.vae.logistic_normal import LogisticNormal
from vampire.predictors import VampirePredictor
from tqdm import tqdm
from scipy import sparse
import json


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)


class VampireModel(object):

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
        if not os.path.exists(data_directory):
            raise FileNotFoundError(f"{data_directory} does not exist. Did you preprocess your data?")            
        vocab = Vocabulary.from_files(os.path.join(data_directory, 'vocabulary'))
        if not ignore_npmi:
            reference_counts = os.path.join(data_directory, "reference", "ref.npz")
            reference_vocabulary = os.path.join(data_directory, "reference", "ref.vocab.json")
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
        if cuda_device > -1:
            self.model.to(cuda_device)
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
                    train_path,
                    dev_path,
                    vocabulary_path,
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

    def extract_features(self, input_, batch=True, scalar_mix: bool=False):
        if batch:
            results = self.model.predict_batch_json(input_)
        else:
            results = [self.model.predict_json(input_)]
        for output in results:
            if scalar_mix:
                output = (torch.Tensor(output['encoder_layer_0']).unsqueeze(0)
                            + -20 * torch.Tensor(output['encoder_layer_1']).unsqueeze(0)
                            + torch.Tensor(output['theta']).unsqueeze(0))
            yield output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden-dim', type=int, default=81)
    parser.add_argument('--kld-clamp', type=int, default=10000)
    parser.add_argument('--train-file', type=str)
    parser.add_argument('--dev-file', type=str, required=False)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--serialization-dir', type=str)
    parser.add_argument('--seed', type=int, default=np.random.randint(0,10000000))
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--vocab-size', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)

    args = parser.parse_args()
    
    # VampireManager.pretokenize(input_file=args.train_file,
    #                            output_file=args.train_file + ".tok.jsonl",
    #                            tokenizer="spacy",
    #                            num_workers=20,
    #                            worker_tqdms=20)
    # VampireManager.pretokenize(input_file=args.dev_file,
    #                         output_file=args.dev_file + ".tok.jsonl",
    #                         tokenizer="spacy",
    #                         num_workers=20,
    #                         worker_tqdms=20)        
    VampireModel.preprocess_data(train_path=args.train_file + ".tok.jsonl",
                                   dev_path=args.dev_file + ".tok.jsonl",
                                   serialization_dir=args.data_dir,
                                   tfidf=True, 
                                   vocab_size=args.vocab_size)                               
    manager = VampireModel.from_params(args.data_dir,
                                       args.kld_clamp,
                                       args.hidden_dim,
                                       args.vocab_size,
                                       ignore_npmi=False)
    manager.fit(args.data_dir,
                args.serialization_dir,
                seed=args.seed,
                cuda_device=args.device)
