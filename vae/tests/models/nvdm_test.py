# pylint: disable=no-self-use,invalid-name
import numpy as np
import re
import copy
from numpy.testing import assert_almost_equal
from vae.models import nvdm
from vae.common.testing.test_case import VAETestCase
from vae.data.tokenizers import regex_and_stopword_filter
from vae.common.allennlp_bridge import VocabularyBGDumper
from allennlp.common.testing import ModelTestCase
from allennlp.commands.train import train_model_from_file
from allennlp.common import Params
from allennlp.data import DataIterator, DatasetReader, Vocabulary
from allennlp.data.dataset import Batch
from allennlp.models import Model, load_archive
from numpy.testing import assert_allclose
import string

class TestNVDM(ModelTestCase):
    def setUp(self):
        super(TestNVDM, self).setUp()
        self.set_up_model(VAETestCase.FIXTURES_ROOT / 'nvdm' / 'experiment.json',
                          VAETestCase.FIXTURES_ROOT / "imdb"  / "train.jsonl")

    
    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)


    def test_dataset_reader_works(self):
        save_dir = self.TEST_DIR / "save_and_load_test"
        archive_file = save_dir / "model.tar.gz"
        model = train_model_from_file(self.param_file, save_dir)
        params = Params.from_file(self.param_file)
        reader = DatasetReader.from_params(params['dataset_reader'])

        # Need to duplicate params because Iterator.from_params will consume.
        iterator_params = params['iterator']
        iterator_params2 = Params(copy.deepcopy(iterator_params.as_dict()))

        iterator = DataIterator.from_params(iterator_params)
        iterator2 = DataIterator.from_params(iterator_params2)

        model_dataset = reader.read(params['validation_data_path'])
        iterator.index_with(model.vocab)
        model_batch = next(iterator(model_dataset, shuffle=False))
        
        # make sure all tokens len > 3
        assert (np.array([len(x) for x in model.vocab._token_to_index['vae']]) > 3).all()
        # make sure no digits in text
        assert (np.array([bool(re.search(r'\d', x)) for x in model.vocab._token_to_index['vae']]) == False).all()

        # make sure no digits in text
        assert (np.array([bool(re.search(r'\d', x)) for x in model.vocab._token_to_index['vae']]) == False).all()
        
        def check_punc(token):
            if token == '@@PADDING@@' or token == '@@UNKNOWN@@':
                return False
            puncs = list(string.punctuation)
            puncs.remove('-')
            if len(set(token).intersection(set(puncs))) > 0:
                return True
            else:
                return False

        # make sure no punctuation in text
        assert (np.array([check_punc(x) for x in model.vocab._token_to_index['vae']]) == False).all()

