DATASETS = {
        "imdb": {
                "train": "s3://suching-dev/final-datasets/imdb/train_pretokenized.jsonl",
                "dev": "s3://suching-dev/final-datasets/imdb/dev_pretokenized.jsonl",
                "test": "s3://suching-dev/final-datasets/imdb/test_pretokenized.jsonl",
                "unlabeled": "s3://suching-dev/final-datasets/imdb/unlabeled_pretokenized.jsonl",
                "reference_counts":  "s3://suching-dev/final-datasets/imdb/valid_npmi_reference/train.npz",
                "reference_vocabulary":  "s3://suching-dev/final-datasets/imdb/valid_npmi_reference/train.vocab.json",
                "stopword_path": "s3://suching-dev/stopwords/snowball_stopwords.txt"
        }
}
