# pylint: disable=no-self-use,invalid-name,missing-docstring
import os
from shutil import rmtree

import pandas as pd

from scripts.generate_data import run, split_data
from vampire.common.testing.test_case import VAETestCase


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
            output_dir=out_dir)

        assert os.path.exists(out_dir)

        full_dev = pd.read_json(os.path.join(out_dir, "dev.jsonl"), lines=True)
        full_test = pd.read_json(os.path.join(out_dir, "test.jsonl"), lines=True)
        full_train = pd.read_json(os.path.join(out_dir, "train.jsonl"), lines=True)
        unlabeled = pd.read_json(os.path.join(out_dir, "unlabeled.jsonl"), lines=True)

        assert full_dev.shape[0] == 2
        assert full_train.shape[0] == 1
        assert full_test.shape[0] == 3
        assert unlabeled.shape[0] == 3
        assert not full_dev.text.isin(full_train.text).any()

        rmtree(out_dir)
