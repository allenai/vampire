# pylint: disable=no-self-use,invalid-name,unused-import
from allennlp.commands.train import train_model_from_file
from allennlp.common.testing import ModelTestCase

from vae.common.allennlp_bridge import VocabularyBGDumper
from vae.common.testing.test_case import VAETestCase
from vae.data.dataset_readers import SemiSupervisedTextClassificationJsonReader
from vae.data.tokenizers import regex_and_stopword_filter
from vae.models import unsupervised


class TestNVDM(ModelTestCase):
    def setUp(self):
        super(TestNVDM, self).setUp()

    def test_model_can_train_save_and_load_logistic(self):
        self.set_up_model(VAETestCase.FIXTURES_ROOT / 'nvdm' / 'experiment_logistic.json',
                          VAETestCase.FIXTURES_ROOT / "imdb" / "train.jsonl")
        self.ensure_model_can_train_save_and_load(self.param_file)
