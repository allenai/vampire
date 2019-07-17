# pylint: disable=no-self-use,invalid-name
import torch
import numpy as np
from allennlp.common import Params
from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch
from vampire.common.testing.test_case import VAETestCase
from vampire.modules.token_embedders import VampireTokenEmbedder


class TestVampireTokenEmbedder(ModelTestCase):
    def setUp(self):
        super().setUp()

    def test_forward_works_with_encoder_output_and_projection(self):
        params = Params({
                'model_archive': VAETestCase.FIXTURES_ROOT / 'vae' / 'model.tar.gz',
                'background_frequency': VAETestCase.FIXTURES_ROOT / 'imdb' / 'vampire.bgfreq',
                'device': -1,
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
        embedding_layer = VampireTokenEmbedder.from_params(vocab=None, params=params)
        assert embedding_layer.get_output_dim() == 20
        input_tensor = torch.LongTensor([word1, word2])
        embedded = embedding_layer(input_tensor).data.numpy()
        assert embedded.shape == (2, 20)

    def test_forward_encoder_output_with_expansion_works(self):
        params = Params({
                'model_archive': VAETestCase.FIXTURES_ROOT / 'vae' / 'model.tar.gz',
                'background_frequency': VAETestCase.FIXTURES_ROOT / 'imdb' / 'vampire.bgfreq',
                'device': -1,
                "expand_dim": True,
                "dropout": 0.0
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
        embedding_layer = VampireTokenEmbedder.from_params(vocab=None, params=params)
        input_tensor = torch.LongTensor([word1, word2])
        expected_vectors = embedding_layer._vae(input_tensor)['vae_representation'].detach().data.numpy()
        embedded = embedding_layer(input_tensor).detach().data.numpy()
        for row in range(input_tensor.shape[0]):
            for col in range(input_tensor.shape[1]):
                np.testing.assert_allclose(embedded[row, col, :], expected_vectors[row, :])
    
    def test_projection_works_with_encoder_weight_representations(self):
        params = Params({
                'model_archive': VAETestCase.FIXTURES_ROOT / 'vae' / 'model.tar.gz',
                'background_frequency': VAETestCase.FIXTURES_ROOT / 'imdb' / 'vampire.bgfreq',
                'device': -1,
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
        embedding_layer = VampireTokenEmbedder.from_params(vocab=None, params=params)
        assert embedding_layer.get_output_dim() == 20
        input_tensor = torch.LongTensor([word1, word2])
        embedded = embedding_layer(input_tensor).data.numpy()
        assert embedded.shape == (2, 50, 20)
    
    def test_forward_works_with_encoder_weight_and_projection(self):
        params = Params({
                'model_archive': VAETestCase.FIXTURES_ROOT / 'vae' / 'model.tar.gz',
                'background_frequency': VAETestCase.FIXTURES_ROOT / 'imdb' / 'vampire.bgfreq',
                'device': -1,
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
        embedding_layer = VampireTokenEmbedder.from_params(vocab=None, params=params)
        assert embedding_layer.get_output_dim() == 20
        input_tensor = torch.LongTensor([word1, word2])
        embedded = embedding_layer(input_tensor).data.numpy()
        assert embedded.shape == (2, 50, 20)
    
    def test_forward_works_with_encoder_output_expand_and_projection(self):
        params = Params({
                'model_archive': VAETestCase.FIXTURES_ROOT / 'vae' / 'model.tar.gz',
                'background_frequency': VAETestCase.FIXTURES_ROOT / 'imdb' / 'vampire.bgfreq',
                'device': -1,
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
        embedding_layer = VampireTokenEmbedder.from_params(vocab=None, params=params)
        assert embedding_layer.get_output_dim() == 20
        input_tensor = torch.LongTensor([word1, word2])
        embedded = embedding_layer(input_tensor).data.numpy()
        assert embedded.shape == (2, 50, 20)