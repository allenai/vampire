from bin.preprocess import clean_text
import pandas as pd
import os
from common.testing.test_case import VAETestCase
from shutil import rmtree

class TestPreprocessing(VAETestCase):

    def test_cleans_text_properly(self):
        data_dir = self.FIXTURES_ROOT / "imdb"
        out_dir = self.TEST_DIR / "imdb"
        train_infile = data_dir / "train.jsonl"        
        dev_infile = data_dir / "dev.jsonl"
        test_infile = data_dir / "test.jsonl"
        test = clean_text("This is a test", strip_html=False, lower=True, keep_emails=False, keep_at_mentions=False)
        assert test == "this is a test"
