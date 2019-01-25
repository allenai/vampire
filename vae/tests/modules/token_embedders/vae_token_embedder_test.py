# pylint: disable=no-self-use,invalid-name
import filecmp
import json
import os
import pathlib
import tarfile
import torch
from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch
from allennlp.data import Vocabulary
from vae.common.testing.test_case import VAETestCase
from vae.models.baselines import logistic_regression
from vae.modules.token_embedders import VAETokenEmbedder
from vae.data.dataset_readers import SemiSupervisedTextClassificationJsonReader
from vae.common import allennlp_bridge

class TestVAETokenEmbedder(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(VAETestCase.FIXTURES_ROOT / 'logistic_regression_vae' / 'experiment.json',
                          VAETestCase.FIXTURES_ROOT / "imdb" / "full" / "train_raw.jsonl")

    def test_clf_with_vae_token_embedder_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_clf_with_vae_token_embedder_forward_pass_runs_correctly(self):
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


    # def test_vocab_extension_attempt_does_not_give_error(self):
    #     # It shouldn't give error if TokenEmbedder does not extend the method `extend_vocab`

    #     params = Params({'options_file': self.FIXTURES_ROOT / 'elmo' / 'options.json',
    #                      'weight_file': self.FIXTURES_ROOT / 'elmo' / 'lm_weights.hdf5'})
    #     embedding_layer = ElmoTokenEmbedder.from_params(vocab=None, params=params)

    #     vocab = Vocabulary()
    #     vocab.add_token_to_namespace('word1')
    #     vocab.add_token_to_namespace('word2')

    #     # This should just pass and be no-op
    #     embedding_layer.extend_vocab(vocab)