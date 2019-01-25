# pylint: disable=no-self-use,invalid-name
import torch
from allennlp.common import Params
from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch
from vae.common.testing.test_case import VAETestCase
from vae.modules.token_embedders import VAETokenEmbedder
from vae.models.baselines import (logistic_regression,
                                  seq2seq_classifier,
                                  seq2vec_classifier)

class TestVAETokenEmbedder(ModelTestCase):
    def setUp(self):
        super().setUp()

    def test_logistic_regression_clf_with_vae_token_embedder_can_train_save_and_load(self):
        self.set_up_model(VAETestCase.FIXTURES_ROOT / 'logistic_regression_vae' / 'experiment.json',
                          VAETestCase.FIXTURES_ROOT / "imdb" / "full" / "train_raw.jsonl")
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_seq2seq_clf_with_vae_token_embedder_can_train_save_and_load(self):
        self.set_up_model(VAETestCase.FIXTURES_ROOT / 'seq2seq_vae' / 'experiment.json',
                          VAETestCase.FIXTURES_ROOT / "imdb" / "full" / "train_raw.jsonl")
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_seq2vec_clf_with_vae_token_embedder_can_train_save_and_load(self):
        self.set_up_model(VAETestCase.FIXTURES_ROOT / 'seq2vec_vae' / 'experiment.json',
                          VAETestCase.FIXTURES_ROOT / "imdb" / "full" / "train_raw.jsonl")
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_logistic_regression_clf_with_vae_token_embedder_forward_pass_runs_correctly(self):
        self.set_up_model(VAETestCase.FIXTURES_ROOT / 'logistic_regression_vae' / 'experiment.json',
                          VAETestCase.FIXTURES_ROOT / "imdb" / "full" / "train_raw.jsonl")
        dataset = Batch(self.instances)
        dataset.index_instances(self.vocab)
        training_tensors = dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        assert output_dict['label_logits'].shape == (3, 2)
        assert output_dict['label_probs'].shape == (3, 2)
        assert output_dict['loss']

    def test_forward_works_with_projection_layer(self):
        params = Params({
                'model_archive': VAETestCase.FIXTURES_ROOT / 'vae' / 'model.tar.gz',
                'background_frequency': VAETestCase.FIXTURES_ROOT / 'vae' / 'vocabulary' / 'vae.bgfreq.json',
                'projection_dim': 20
                })
        word1 = [0] * 50
        word2 = [0] * 50
        word1[0] = 6
        word1[1] = 5
        word1[2] = 4
        word1[3] = 3
        word2[0] = 3
        word2[1] = 2
        word2[2] = 1
        word2[3] = 0
        embedding_layer = VAETokenEmbedder.from_params(vocab=None, params=params)
        assert embedding_layer.get_output_dim() == 20
        input_tensor = torch.LongTensor([word1, word2])
        embedded = embedding_layer(input_tensor).data.numpy()
        assert embedded.shape == (2, 20)
    
    def test_forward_works_with_non_default_representations(self):
        params = Params({
                'model_archive': VAETestCase.FIXTURES_ROOT / 'vae' / 'model.tar.gz',
                'background_frequency': VAETestCase.FIXTURES_ROOT / 'vae' / 'vocabulary' / 'vae.bgfreq.json',
                'representation': 'encoder_weights',
                'projection_dim': 20
                })
        word1 = [0] * 50
        word2 = [0] * 50
        word1[0] = 6
        word1[1] = 5
        word1[2] = 4
        word1[3] = 3
        word2[0] = 3
        word2[1] = 2
        word2[2] = 1
        word2[3] = 0
        embedding_layer = VAETokenEmbedder.from_params(vocab=None, params=params)
        assert embedding_layer.get_output_dim() == 20
        input_tensor = torch.LongTensor([word1, word2])
        embedded = embedding_layer(input_tensor).data.numpy()
        assert embedded.shape == (2, 50, 20)

    def test_forward_works_with_combine(self):
        params = Params({
                'model_archive': VAETestCase.FIXTURES_ROOT / 'vae' / 'model.tar.gz',
                'background_frequency': VAETestCase.FIXTURES_ROOT / 'vae' / 'vocabulary' / 'vae.bgfreq.json',
                'representation': 'encoder_weights',
                'projection_dim': 20,
                'combine': True
                })
        word1 = [0] * 50
        word2 = [0] * 50
        word1[0] = 6
        word1[1] = 5
        word1[2] = 4
        word1[3] = 3
        word2[0] = 3
        word2[1] = 2
        word2[2] = 1
        word2[3] = 0
        embedding_layer = VAETokenEmbedder.from_params(vocab=None, params=params)
        assert embedding_layer.get_output_dim() == 20
        input_tensor = torch.LongTensor([word1, word2])
        embedded = embedding_layer(input_tensor).data.numpy()
        assert embedded.shape == (2, 20)
    
    def test_forward_works_with_expand(self):
        params = Params({
                'model_archive': VAETestCase.FIXTURES_ROOT / 'vae' / 'model.tar.gz',
                'background_frequency': VAETestCase.FIXTURES_ROOT / 'vae' / 'vocabulary' / 'vae.bgfreq.json',
                'representation': 'encoder_weights',
                'projection_dim': 20,
                'expand_dim': True
                })
        word1 = [0] * 50
        word2 = [0] * 50
        word1[0] = 6
        word1[1] = 5
        word1[2] = 4
        word1[3] = 3
        word2[0] = 3
        word2[1] = 2
        word2[2] = 1
        word2[3] = 0
        embedding_layer = VAETokenEmbedder.from_params(vocab=None, params=params)
        assert embedding_layer.get_output_dim() == 20
        input_tensor = torch.LongTensor([word1, word2])
        embedded = embedding_layer(input_tensor).data.numpy()
        assert embedded.shape == (2, 50, 20)