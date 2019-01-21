from bin.generate_data import run, split_data
import pandas as pd
import os
from common.testing.test_case import VAETestCase
from shutil import rmtree

class TestGenerateData(VAETestCase):

    def test_splits_data_correctly(self):
        df = pd.DataFrame({'0': [0] * 4, '1': [1] * 4})
        df, other = split_data(df, 2)
        expected_df = pd.DataFrame({'0': [0] * 2, '1': [1] * 2})
        expected_other = pd.DataFrame({'0': [0] * 2, '1': [1] * 2})
        pd.testing.assert_frame_equal(df, expected_df)
        pd.testing.assert_frame_equal(other, expected_other)

    def test_runs_imdb_properly(self):
        data_dir = self.FIXTURES_ROOT / "imdb"
        out_dir = self.TEST_DIR / "imdb"
        
        run(split_dev=2,
            data_dir=data_dir,
            output_dir=out_dir,
            subsamples=[1])
        
        assert os.path.exists(out_dir)
        dirs = os.listdir(out_dir)
        assert len(dirs) == 3
        assert "1" in dirs
        assert "unlabeled" in dirs
        assert "full" in dirs

        full_dir = os.path.join(out_dir, "full")
        unlabeled_dir = os.path.join(out_dir, "unlabeled")
        sample_dir = os.path.join(out_dir, "1")
        full_dev = pd.read_json(os.path.join(full_dir, "dev_raw.jsonl"), lines=True)
        full_test = pd.read_json(os.path.join(full_dir, "test_raw.jsonl"), lines=True)
        full_train = pd.read_json(os.path.join(full_dir, "train_raw.jsonl"), lines=True)
        train_sample = pd.read_json(os.path.join(sample_dir, "train_raw.jsonl"), lines=True)
        unlabeled = pd.read_json(os.path.join(unlabeled_dir, "train_raw.jsonl"), lines=True)

        assert full_dev.shape[0] == 2
        assert full_train.shape[0] == 1
        assert train_sample.shape[0] == 1
        assert full_test.shape[0] == 3
        assert unlabeled.shape[0] == 3
        assert not full_dev.text.isin(full_train.text).any()
        assert train_sample.text.isin(full_train.text).all()

        rmtree(out_dir)