from vampire.api import VampireTokenizer
import pytest
from vampire.common.testing import VAETestCase
import numpy as np
from vampire.common.util import load_sparse


class TestTokenizerAPI(VAETestCase):

    def test_spacy_tokenizer(self):
        archive_path = self.FIXTURES_ROOT / "vae" / "model.tar.gz"
        input_file = self.FIXTURES_ROOT / "ag" / "train.jsonl"
        output_file = self.TEST_DIR / "train.tok.jsonl"
        tokenizer = VampireTokenizer(tokenizer="spacy")
        tokenizer.pretokenize(input_file, output_file, is_json=True, lower=True)
        assert output_file.exists()
        with open(output_file, 'r') as f:
            z = f.readlines()
        assert z[0] == '{"label": 4, "text": "extended collaboration could result in chips built on 32-nanometer technology .", "headline": "IBM, AMD Work to Shrink Chips", "id": "id31216"}\n'
    