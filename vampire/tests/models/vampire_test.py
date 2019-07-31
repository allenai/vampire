# pylint: disable=no-self-use,invalid-name,unused-import
import numpy as np
from allennlp.commands.train import train_model_from_file
from allennlp.common.testing import ModelTestCase

from vampire.common.allennlp_bridge import ExtendedVocabulary
from vampire.common.testing.test_case import VAETestCase
from vampire.data.dataset_readers import VampireReader
from vampire.models import VAMPIRE


class TestVampire(ModelTestCase):
    def setUp(self):
        super(TestVampire, self).setUp()
        self.set_up_model(VAETestCase.FIXTURES_ROOT / 'unsupervised' / 'experiment.json',
                          VAETestCase.FIXTURES_ROOT / "imdb" / "train.npz")

    def test_model_can_train_save_and_load_unsupervised(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_npmi_computed_correctly(self):
        save_dir = self.TEST_DIR / "save_and_load_test"
        model = train_model_from_file(self.param_file, save_dir, overrides="")

        topics = [(1, ["great", "movie", "film", "amazing", "wow", "best", "ridiculous", "ever", "good", "incredible", "positive"]),
                  (2, ["bad", "film", "worst", "negative", "movie", "ever", "not", "any", "gross", "boring"])]
        npmi = model.compute_npmi(topics, num_words=10)

        ref_vocab = model._ref_vocab
        ref_counts = model._ref_count_mat

        vocab_index = dict(zip(ref_vocab, range(len(ref_vocab))))
        n_docs, _ = ref_counts.shape

        npmi_means = []
        for topic in topics:
            words = topic[1]
            npmi_vals = []
            for word_i, word1 in enumerate(words[:10]):
                if word1 in vocab_index:
                    index1 = vocab_index[word1]
                else:
                    index1 = None
                for word2 in words[word_i+1:10]:
                    if word2 in vocab_index:
                        index2 = vocab_index[word2]
                    else:
                        index2 = None
                    if index1 is None or index2 is None:
                        _npmi = 0.0
                    else:
                        col1 = np.array(ref_counts[:, index1].todense() > 0, dtype=int)
                        col2 = np.array(ref_counts[:, index2].todense() > 0, dtype=int)
                        sum1 = col1.sum()
                        sum2 = col2.sum()
                        interaction = np.sum(col1 * col2)
                        if interaction == 0:
                            assert model._npmi_numerator[index1, index2] == 0.0 and model._npmi_denominator[index1, index2] == 0.0
                            _npmi = 0.0
                        else:
                            assert model._ref_interaction[index1, index2] == np.log10(interaction)
                            assert model._ref_doc_sum[index1] == sum1
                            assert model._ref_doc_sum[index2] == sum2
                            expected_numerator = np.log10(n_docs) + np.log10(interaction) - np.log10(sum1) - np.log10(sum2)
                            numerator = np.log10(model.n_docs) + model._npmi_numerator[index1, index2]
                            assert np.isclose(expected_numerator, numerator)
                            expected_denominator = np.log10(n_docs) - np.log10(interaction)
                            denominator = np.log10(model.n_docs) - model._npmi_denominator[index1, index2]
                            assert np.isclose(expected_denominator, denominator)
                            _npmi = expected_numerator / expected_denominator
                    npmi_vals.append(_npmi)
            npmi_means.append(np.mean(npmi_vals))
        assert np.isclose(npmi, np.mean(npmi_means))
