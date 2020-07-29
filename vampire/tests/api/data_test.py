from vampire.api import preprocess_data, transform_text
import pytest
from vampire.common.testing import VAETestCase
import numpy as np

class TestDataProcessing(VAETestCase):

    def test_preprocess_data_tfidf_works(self):
        train_path = self.FIXTURES_ROOT / "ag" / "train.jsonl"
        dev_path = self.FIXTURES_ROOT / "ag" / "dev.jsonl"
        serialization_dir = self.TEST_DIR / "preprocessed_data"
        preprocess_data(train_path, dev_path, serialization_dir, tfidf=True, vocab_size=300)
        z = np.load(serialization_dir / 'train.npz')['data']
        assert serialization_dir.exists()
        assert (serialization_dir / 'vocabulary' / 'vampire.txt').exists()
        
        np.testing.assert_almost_equal(z, np.array([0.37796447, 0.37796447, 0.37796447, 0.37796447, 0.37796447,
                                                    0.37796447, 0.37796447, 0.23787672, 0.23787672, 0.23787672,
                                                    0.19506231, 0.23787672, 0.23787672, 0.23787672, 0.23787672,
                                                    0.23787672, 0.23787672, 0.23787672, 0.23787672, 0.23787672,
                                                    0.23787672, 0.23787672, 0.23787672, 0.23787672, 0.23787672,
                                                    0.20132333, 0.20132333, 0.20132333, 0.40264665, 0.16508801,
                                                    0.20132333, 0.20132333, 0.20132333, 0.20132333, 0.20132333,
                                                    0.20132333, 0.20132333, 0.20132333, 0.20132333, 0.20132333,
                                                    0.20132333, 0.20132333, 0.20132333, 0.20132333, 0.20132333,
                                                    0.20132333, 0.20132333]))

    def test_preprocess_data_no_tfidf_works(self):
        train_path = self.FIXTURES_ROOT / "ag" / "train.jsonl"
        dev_path = self.FIXTURES_ROOT / "ag" / "dev.jsonl"
        serialization_dir = self.TEST_DIR / "preprocessed_data"
        preprocess_data(train_path, dev_path, serialization_dir, tfidf=False, vocab_size=5)
        z = np.load(serialization_dir / 'train.npz')['data']
        with open(serialization_dir / 'vocabulary' / 'vampire.txt', 'r') as f:
            words = f.readlines()
        assert words == ['@@UNKNOWN@@\n', 'american\n', 'batteries\n', 'new\n', 'second\n', 'security\n']
        np.testing.assert_almost_equal(z, np.array([1, 1, 1, 2]))

    def test_transform_text(self):
        input_file = self.FIXTURES_ROOT / 'ag' / 'train.jsonl'
        vocabulary_path = self.FIXTURES_ROOT / 'vae' / 'vocabulary' / 'vampire.txt'
        serialization_dir = self.TEST_DIR / 'shards'
        transform_text(input_file, vocabulary_path, tfidf=True, serialization_dir=serialization_dir, shard=True, num_shards=3)
        assert serialization_dir.exists()
        assert len(list(serialization_dir.iterdir())) == 3