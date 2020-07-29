# pylint: disable=no-self-use,invalid-name
import pytest
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from vampire.common.testing import VAETestCase
from allennlp.common.util import ensure_list, prepare_environment

from vampire.data.dataset_readers import SemiSupervisedTextClassificationJsonReader


class TestSemiSupervisedTextClassificationJsonReader(VAETestCase):

    def test_read_from_file(self):
        reader = SemiSupervisedTextClassificationJsonReader()
        ag_path = self.FIXTURES_ROOT / "ag" / "train.jsonl"
        instances = reader.read(ag_path)
        instances = ensure_list(instances)

        instance1 = {"tokens": ["Extended", "collaboration", "could", "result", "in", "chips", "built", "on", "32-nanometer", "technology", "."],
                     "label": "4"}
        instance2 = {"tokens": ['BAGHDAD', '(', 'Reuters', ')', '-', 'Al', 'Qaeda', '-', 'linked', 'militants', 'said', 'they', 'had', 
                                'killed', 'a', 'second', 'American', 'hostage', 'and', 'threatened', 'to', 'kill', 'a', 'Briton', 
                                'unless', 'women', 'prisoners', 'in', 'Iraq', 'were', 'freed', '.'],
                     "label": "1"}
        instance3 = {"tokens": ['AOL', 'is', 'getting', 'serious', 'about', 'security', '-', 'if', 'you', 'pay', 'them', 
                                'an', 'extra', '\\$1.95', 'to', '\\$4.95', 'a', 'month', '.', 'Grandma', '#', '39;s', 
                                'favorite', 'ISP', 'has', 'teamed', 'with', 'RSA', 'Security', 'to', 'offer', 'an', 
                                'optional', 'login', 'service', 'that', 'provides', 'a', 'second', 'layer', 'of', 
                                'authentication', 'to', 'prevent', 'identity', 'theft', '.'],
                     "label": "4"}
        
        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["tokens"].tokens] == instance1["tokens"]
        assert fields["label"].label == instance1["label"]
        fields = instances[1].fields
        assert [t.text for t in fields["tokens"].tokens] == instance2["tokens"]
        assert fields["label"].label == instance2["label"]
        fields = instances[2].fields
        assert [t.text for t in fields["tokens"].tokens] == instance3["tokens"]
        assert fields["label"].label == instance3["label"]

    def test_read_from_file_and_truncates_properly(self):

        reader = SemiSupervisedTextClassificationJsonReader(max_sequence_length=5)
        ag_path = self.FIXTURES_ROOT / "ag" / "train.jsonl"
        instances = reader.read(ag_path)
        instances = ensure_list(instances)

        instance1 = {"tokens": ['Extended', 'collaboration', 'could', 'result', 'in'],
                     "label": "4"}
        instance2 = {"tokens": ['BAGHDAD', '(', 'Reuters', ')', '-'],
                     "label": "1"}
        instance3 = {"tokens": ['AOL', 'is', 'getting', 'serious', 'about'],
                     "label": "4"}

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["tokens"].tokens] == instance1["tokens"]
        assert fields["label"].label == instance1["label"]
        fields = instances[1].fields
        assert [t.text for t in fields["tokens"].tokens] == instance2["tokens"]
        assert fields["label"].label == instance2["label"]
        fields = instances[2].fields
        assert [t.text for t in fields["tokens"].tokens] == instance3["tokens"]
        assert fields["label"].label == instance3["label"]

    def test_samples_properly(self):
        reader = SemiSupervisedTextClassificationJsonReader(sample=1, max_sequence_length=5)
        ag_path = self.FIXTURES_ROOT / "ag" / "train.jsonl"
        params = Params({"random_seed": 5, "numpy_seed": 5, "pytorch_seed": 5})
        prepare_environment(params)
        instances = reader.read(ag_path)
        instances = ensure_list(instances)
        instance = {"tokens": ['BAGHDAD', '(', 'Reuters', ')', '-'],
                    "label": "1"}
        assert len(instances) == 1
        fields = instances[0].fields
        assert [t.text for t in fields["tokens"].tokens] == instance["tokens"]
        assert fields["label"].label == instance["label"]

    def test_sampling_fails_when_sample_size_larger_than_file_size(self):
        reader = SemiSupervisedTextClassificationJsonReader(sample=10, max_sequence_length=5)
        ag_path = self.FIXTURES_ROOT / "ag" / "train.jsonl"
        params = Params({"random_seed": 5, "numpy_seed": 5, "pytorch_seed": 5})
        prepare_environment(params)
        pytest.raises(ConfigurationError, reader.read, ag_path)

    def test_samples_according_to_seed_properly(self):

        reader1 = SemiSupervisedTextClassificationJsonReader(sample=2, max_sequence_length=5)
        reader2 = SemiSupervisedTextClassificationJsonReader(sample=2, max_sequence_length=5)
        reader3 = SemiSupervisedTextClassificationJsonReader(sample=2, max_sequence_length=5)

        imdb_path = self.FIXTURES_ROOT / "ag" / "train.jsonl"
        params = Params({"random_seed": 5, "numpy_seed": 5, "pytorch_seed": 5})
        prepare_environment(params)
        instances1 = reader1.read(imdb_path)
        params = Params({"random_seed": 2, "numpy_seed": 2, "pytorch_seed": 2})
        prepare_environment(params)
        instances2 = reader2.read(imdb_path)
        params = Params({"random_seed": 5, "numpy_seed": 5, "pytorch_seed": 5})
        prepare_environment(params)
        instances3 = reader3.read(imdb_path)
        fields1 = [i.fields for i in instances1]
        fields2 = [i.fields for i in instances2]
        fields3 = [i.fields for i in instances3]
        tokens1 = [f['tokens'].tokens for f in fields1]
        tokens2 = [f['tokens'].tokens for f in fields2]
        tokens3 = [f['tokens'].tokens for f in fields3]
        text1 = [[t.text for t in doc] for doc in tokens1]
        text2 = [[t.text for t in doc] for doc in tokens2]
        text3 = [[t.text for t in doc] for doc in tokens3]
        assert text1 != text2
        assert text1 == text3

    def test_ignores_label_properly(self):

        imdb_labeled_path = self.FIXTURES_ROOT / "ag" / "train.jsonl"
        reader = SemiSupervisedTextClassificationJsonReader(ignore_labels=True)
        instances = reader.read(imdb_labeled_path)
        instances = ensure_list(instances)
        fields = [i.fields for i in instances]
        labels = [f.get('label') for f in fields]
        assert labels == [None] * 3
