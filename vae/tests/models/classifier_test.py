import numpy as np
import torch
from allennlp.common import Params
from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch

from vae.common.testing.test_case import VAETestCase
from vae.models import classifier
from vae.modules.token_embedders import VAETokenEmbedder


class TestClassifiers(ModelTestCase):
    def setUp(self):
        super().setUp()

    def test_logistic_regression_clf_with_vae_token_embedder_can_train_save_and_load(self):
        self.set_up_model(VAETestCase.FIXTURES_ROOT / 'classifier' / 'experiment_logistic_regression.json',
                          VAETestCase.FIXTURES_ROOT / "imdb" / "train.jsonl")
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_logistic_regression_clf_batch_predictions_are_consistent(self):
        self.set_up_model(VAETestCase.FIXTURES_ROOT / 'classifier' / 'experiment_logistic_regression.json',
                          VAETestCase.FIXTURES_ROOT / "imdb" / "train.jsonl")          
        self.ensure_batch_predictions_are_consistent()

    def test_seq2seq_clf_with_vae_token_embedder_can_train_save_and_load(self):
        self.set_up_model(VAETestCase.FIXTURES_ROOT / 'classifier' / 'experiment_seq2seq.json',
                          VAETestCase.FIXTURES_ROOT / "imdb" / "train.jsonl")
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_seq2seq_clf_with_vae_token_embedder_batch_predictions_are_consistent(self): 
        self.set_up_model(VAETestCase.FIXTURES_ROOT / 'classifier' / 'experiment_seq2seq.json',
                          VAETestCase.FIXTURES_ROOT / "imdb" / "train.jsonl")          
        self.ensure_batch_predictions_are_consistent()

    def test_seq2vec_clf_with_vae_token_embedder_can_train_save_and_load(self):
        self.set_up_model(VAETestCase.FIXTURES_ROOT / 'classifier' / 'experiment_seq2vec.json',
                          VAETestCase.FIXTURES_ROOT / "imdb" / "train.jsonl")
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_logistic_regression_clf_with_vae_token_embedder_forward_pass_runs_correctly(self):
        self.set_up_model(VAETestCase.FIXTURES_ROOT / 'classifier' / 'experiment_logistic_regression.json',
                          VAETestCase.FIXTURES_ROOT / "imdb" / "train.jsonl")
        dataset = Batch(self.instances)
        dataset.index_instances(self.vocab)
        training_tensors = dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        assert output_dict['label_logits'].shape == (3, 2)
        assert output_dict['label_probs'].shape == (3, 2)
        assert output_dict['loss']

    def test_seq2seq_clf_with_vae_token_embedder_forward_pass_runs_correctly(self):
        self.set_up_model(VAETestCase.FIXTURES_ROOT / 'classifier' / 'experiment_seq2seq.json',
                          VAETestCase.FIXTURES_ROOT / "imdb" / "train.jsonl")
        dataset = Batch(self.instances)
        dataset.index_instances(self.vocab)
        training_tensors = dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        assert output_dict['label_logits'].shape == (3, 2)
        assert output_dict['label_probs'].shape == (3, 2)
        assert output_dict['loss']

    def test_seq2vec_clf_with_vae_token_embedder_forward_pass_runs_correctly(self):
        self.set_up_model(VAETestCase.FIXTURES_ROOT / 'classifier' / 'experiment_seq2vec.json',
                          VAETestCase.FIXTURES_ROOT / "imdb" / "train.jsonl")
        dataset = Batch(self.instances)
        dataset.index_instances(self.vocab)
        training_tensors = dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        assert output_dict['label_logits'].shape == (3, 2)
        assert output_dict['label_probs'].shape == (3, 2)
        assert output_dict['loss']
