DATASETS = {
    "tone": {
        "train": "s3://suching-dev/processed-datasets/tone/train.jsonl",
        "dev": "s3://suching-dev/processed-datasets/tone/dev.jsonl",
        "test": "s3://suching-dev/processed-datasets/tone/test.jsonl",
        "unlabeled": "s3://suching-dev/processed-datasets/tone/unlabeled.jsonl",
        "reference_counts":  "s3://suching-dev/processed-datasets/tone/valid_npmi_reference/train.npz",
        "reference_vocabulary":  "s3://suching-dev/processed-datasets/tone/valid_npmi_reference/train.vocab.json",
        "stopword_path": "s3://suching-dev/stopwords/snowball_stopwords.txt",
    },
    "imdb": {
        "train": "s3://suching-dev/processed-datasets/imdb/train.jsonl",
        "dev": "s3://suching-dev/processed-datasets/imdb/dev.jsonl",
        "test": "s3://suching-dev/processed-datasets/imdb/test.jsonl",
        "unlabeled": "s3://suching-dev/processed-datasets/imdb/unlabeled.jsonl",
        "reference_counts":  "s3://suching-dev/processed-datasets/imdb/valid_npmi_reference/train.npz",
        "reference_vocabulary":  "s3://suching-dev/processed-datasets/imdb/valid_npmi_reference/train.vocab.json",
        "stopword_path": "s3://suching-dev/stopwords/snowball_stopwords.txt",
    },
    "authorship": {
        "train": "s3://suching-dev/processed-datasets/authorship/train.jsonl",
        "dev": "s3://suching-dev/processed-datasets/authorship/dev.jsonl",
        "test": "s3://suching-dev/processed-datasets/authorship/test.jsonl",
        "unlabeled": "s3://suching-dev/processed-datasets/authorship/unlabeled.jsonl",
        "reference_counts":  "s3://suching-dev/processed-datasets/authorship/valid_npmi_reference/train.npz",
        "reference_vocabulary":  "s3://suching-dev/processed-datasets/authorship/valid_npmi_reference/train.vocab.json",
        "stopword_path": "s3://suching-dev/stopwords/snowball_stopwords.txt",
    },
}
