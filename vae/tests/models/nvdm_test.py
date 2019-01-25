# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import ModelTestCase
from vae.common.testing.test_case import VAETestCase


class TestNVDM(ModelTestCase):
    def setUp(self):
        super(TestNVDM, self).setUp()
        self.set_up_model(VAETestCase.FIXTURES_ROOT / 'nvdm' / 'experiment.json',
                          VAETestCase.FIXTURES_ROOT / "imdb" / "train.jsonl")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
