from bin.preprocess import clean_text, tokenize, generate_vocab, generate_bg_frequency, run
import pandas as pd
import os
from common.testing.test_case import VAETestCase
from shutil import rmtree
from collections import Counter
from common import file_handling as fh
class TestPreprocessing(VAETestCase):

    def test_cleans_text_properly(self):
        test = clean_text("This is a test <p>this is html </p> suchin@ss.com @ssgrn    ",
                          strip_html=True,
                          lower=True,
                          keep_emails=False,
                          keep_at_mentions=False)
        assert test == "this is a test this is html"

    def test_tokenizes_properly(self):
        test = tokenize("This is a_test <p>this is html </p> dog's suchin@ss.com @ssgrn    ",
                        strip_html=True,
                        lower=True,
                        keep_emails=False,
                        keep_at_mentions=False,
                        min_length=3)
        assert test == ['this', 'test', 'this', 'html', 'dogs']
    
    def test_generates_vocab_correctly(self):
        doc_counts = Counter()
        docs = ["this is a test sentence",
                "this is another cool test sentence",
                "allennlp is a cool library",
                "this is a cool sentence"]
        for doc in docs:
            tokens = doc.split()
            doc_counts.update(tokens)

        vocab = generate_vocab(doc_counts=doc_counts, n_items=4, vocab_size=10)
        assert vocab == ['@@UNKNOWN@@', 'a', 'allennlp', 'another', 'cool', 'is', 'library', 'sentence', 'test', 'this']

    def test_generates_bg_freq_properly(self):
        doc_counts = Counter()
        parsed_text = [["this", "is", "a", "test", "sentence"],
                        ["this", "is" , "another" , "cool", "test", "sentence"],
                        ["allennlp", "is" , "a", "cool", "library"],
                        ["this", "is", "a", "cool", "sentence"]]
        bg_freq = generate_bg_frequency(pd.Series(parsed_text))
        expected_freq = {
                            "this": 3 / 21,
                            "allennlp": 1 / 21,
                            "is": 4 / 21,
                            "a": 3 / 21,
                            "cool": 3 / 21,
                            "test": 2 / 21,
                            "sentence" : 3 / 21,
                            "library": 1 / 21,
                            "another": 1 / 21,
                         }
        for key in expected_freq:
            assert expected_freq[key] == bg_freq[key]

    def test_files_preprocessed_correctly(self):
        train = self.FIXTURES_ROOT / "imdb" / "full" / "train_raw.jsonl"
        test = self.FIXTURES_ROOT / "imdb" / "full" / "test_raw.jsonl"
        out_dir = self.TEST_DIR / "imdb"
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        run(train_infile=train,
            test_infile=test,
            dev_infile=None,
            output_dir=out_dir,
            train_prefix="train",
            test_prefix="test",
            dev_prefix="dev",
            min_doc_count=0,
            max_doc_freq=1,
            vocab_size=10,
            stopwords="snowball",
            keep_num=False,
            keep_alphanum=False,
            strip_html=True,
            lower=True,
            min_length=3,
            label_field="category")

        files = os.listdir(out_dir)
        files.sort()
        assert files == ['test.jsonl', 'test.txt', 'train.bgfreq.json', 'train.jsonl', 'train.txt', 'vocabulary']
        
        with open(os.path.join(out_dir, "train.txt")) as f:
            train_txt = f.readlines()

        with open(os.path.join(out_dir, "test.txt")) as f:
            test_txt = f.readlines()

        train_df = pd.read_json(os.path.join(out_dir, "train.jsonl"), lines=True)
        test_df = pd.read_json(os.path.join(out_dir, "test.jsonl"), lines=True)

        train_bgfreq = fh.read_json(os.path.join(out_dir, "train.bgfreq.json"))

        files = os.listdir(os.path.join(out_dir, "vocabulary"))
        files.sort()
        assert files == ['is_labeled.txt', 'labels.txt', 'non_padded_namespaces.txt', 'tokens.txt']

        namespaces = fh.read_text(os.path.join(out_dir, "vocabulary", "non_padded_namespaces.txt"))
        assert namespaces == ['tokens', 'labels', 'is_labeled']

        vocab = fh.read_text(os.path.join(out_dir, "vocabulary", "tokens.txt"))
        assert len(vocab) == 11
        assert "@@UNKNOWN@@" in vocab

        labels = fh.read_text(os.path.join(out_dir, "vocabulary", "labels.txt"))
        assert labels == ["0", "1"]

        is_labeled = fh.read_text(os.path.join(out_dir, "vocabulary", "is_labeled.txt"))
        assert is_labeled == ["0", "1"]

        assert len(train_bgfreq) == len(vocab) - 1