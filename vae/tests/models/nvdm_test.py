# pylint: disable=no-self-use,invalid-name,unused-import
from allennlp.common.testing import ModelTestCase
from allennlp.commands.train import train_model_from_file
from vae.data.dataset_readers import SemiSupervisedTextClassificationJsonReader
from vae.data.tokenizers import regex_and_stopword_filter
from vae.common.allennlp_bridge import VocabularyBGDumper
from vae.models import NVDM
from vae.common.testing.test_case import VAETestCase

class TestNVDM(ModelTestCase):
    def setUp(self):
        super(TestNVDM, self).setUp()
        self.set_up_model(VAETestCase.FIXTURES_ROOT / 'nvdm' / 'experiment.json',
                          VAETestCase.FIXTURES_ROOT / "imdb" / "train.jsonl")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
