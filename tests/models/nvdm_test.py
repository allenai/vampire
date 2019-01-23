# pylint: disable=no-self-use,invalid-name
import numpy
from numpy.testing import assert_almost_equal
from models import nvdm
from common.testing.test_case import VAETestCase
from data.tokenizers import regex_and_stopword_filter
from common.allennlp_bridge import VocabularyBGDumper
from allennlp.common.testing import ModelTestCase


class TestNVDM(ModelTestCase):
    def setUp(self):
        super(TestNVDM, self).setUp()
        self.set_up_model(VAETestCase.FIXTURES_ROOT / 'nvdm' / 'experiment.json',
                          VAETestCase.FIXTURES_ROOT / "imdb"  / "train.jsonl")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
