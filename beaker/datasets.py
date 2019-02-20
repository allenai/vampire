DATASETS = {
    "imdb": {
        "train": "s3://suching-dev/final-datasets/imdb/train.jsonl",
        "dev": "s3://suching-dev/final-datasets/imdb/dev.jsonl",
        "test": "s3://suching-dev/final-datasets/imdb/test.jsonl",
        "unlabeled": "s3://suching-dev/final-datasets/imdb/unlabeled.jsonl",
        "reference_counts":  "s3://suching-dev/final-datasets/imdb/valid_npmi_reference/train.npz",
        "reference_vocabulary":  "s3://suching-dev/final-datasets/imdb/valid_npmi_reference/train.vocab.json",
        "stopword_path": "s3://suching-dev/stopwords/snowball_stopwords.txt",
        "glove": "s3://suching-dev/pretrained-models/glove/imdb/vectors.txt",
        "elmo": {
            "frozen": "s3://allennlp/models/transformer-elmo-2019.01.10.tar.gz",
            "fine-tuned": "s3://suching-dev/pretrained-models/elmo/imdb/model.tar.gz",
            "in-domain": "s3://suching-dev/pretrained-models/in-domain-elmo/imdb/model.tar.gz"
        },
        "bert": {
            "weights": "s3://suching-dev/pretrained-models/bert/imdb/model.tar.gz",
            "vocab": "s3://suching-dev/pretrained-models/bert/imdb/vocab.txt"
        },
        "vae": {
            "model_archive": "s3://suching-dev/pretrained-models/vae_best_npmi/imdb/model.tar.gz",
            "vocab": "s3://suching-dev/pretrained-models/vae_best_npmi/imdb/vae.txt",
            "bg_freq": "s3://suching-dev/pretrained-models/vae_best_npmi/imdb/vae.bgfreq.json"
        }
    },
    "1b": {
        "train": "s3://suching-dev/final-datasets/1b/train.jsonl",
        "test": "s3://suching-dev/final-datasets/1b/test.jsonl",
        "reference_counts":  "s3://suching-dev/final-datasets/1b/test_npmi_reference/train.npz",
        "reference_vocabulary":  "s3://suching-dev/final-datasets/1b/test_npmi_reference/train.vocab.json",
        "stopword_path": "s3://suching-dev/stopwords/snowball_stopwords.txt",
        "glove": "s3://suching-dev/pretrained-models/glove/1b/vectors.txt",
        "elmo": {
            "frozen": "s3://allennlp/models/transformer-elmo-2019.01.10.tar.gz",
            "fine-tuned": "s3://suching-dev/pretrained-models/elmo/1b/model.tar.gz",
            "in-domain": "s3://suching-dev/pretrained-models/in-domain-elmo/1b/model.tar.gz"
        },
        "bert": {
            "weights": "s3://suching-dev/pretrained-models/bert/1b/model.tar.gz",
            "vocab": "s3://suching-dev/pretrained-models/bert/1b/vocab.txt"
        },
         "vae": {
            "model_archive": "s3://suching-dev/pretrained-models/vae_best_npmi/1b/model.tar.gz",
            "vocab": "s3://suching-dev/pretrained-models/vae_best_npmi/1b/vae.txt",
            "bg_freq": "s3://suching-dev/pretrained-models/vae_best_npmi/1b/vae.bgfreq.json"
        }
    },
    "amazon": {
        "train": "s3://suching-dev/final-datasets/amazon/train.jsonl",
        "dev": "s3://suching-dev/final-datasets/amazon/dev.jsonl",
        "test": "s3://suching-dev/final-datasets/amazon/test.jsonl",
        "unlabeled": "s3://suching-dev/final-datasets/amazon/unlabeled.jsonl",
        "reference_counts":  "s3://suching-dev/final-datasets/amazon/valid_npmi_reference/train.npz",
        "reference_vocabulary":  "s3://suching-dev/final-datasets/amazon/valid_npmi_reference/train.vocab.json",
        "stopword_path": "s3://suching-dev/stopwords/snowball_stopwords.txt",
        "glove": "s3://suching-dev/pretrained-models/glove/amazon/vectors.txt",
        "elmo": {
            "frozen": "s3://allennlp/models/transformer-elmo-2019.01.10.tar.gz",
            "fine-tuned": "s3://suching-dev/pretrained-models/elmo/amazon/model.tar.gz",
            "in-domain": "s3://suching-dev/pretrained-models/in-domain-elmo/amazon/model.tar.gz"
        },
        "bert": {
            "weights": "s3://suching-dev/pretrained-models/bert/amazon/model.tar.gz",
            "vocab": "s3://suching-dev/pretrained-models/bert/amazon/vocab.txt"
        },
         "vae": {
            "model_archive": "s3://suching-dev/pretrained-models/vae_best_npmi/amazon/model.tar.gz",
            "vocab": "s3://suching-dev/pretrained-models/vae_best_npmi/amazon/vae.txt",
            "bg_freq": "s3://suching-dev/pretrained-models/vae_best_npmi/amazon/vae.bgfreq.json"
        }

    },
    "yahoo": {
        "train": "s3://suching-dev/final-datasets/yahoo/train.jsonl",
        "dev": "s3://suching-dev/final-datasets/yahoo/dev.jsonl",
        "test": "s3://suching-dev/final-datasets/yahoo/test.jsonl",
        "unlabeled": "s3://suching-dev/final-datasets/yahoo/unlabeled.jsonl",
        "reference_counts":  "s3://suching-dev/final-datasets/yahoo/valid_npmi_reference/train.npz",
        "reference_vocabulary":  "s3://suching-dev/final-datasets/yahoo/valid_npmi_reference/train.vocab.json",
        "stopword_path": "s3://suching-dev/stopwords/snowball_stopwords.txt",
        "glove": "s3://suching-dev/pretrained-models/glove/yahoo/vectors.txt",
        "elmo": {
            "frozen": "s3://allennlp/models/transformer-elmo-2019.01.10.tar.gz",
            "fine-tuned": "s3://suching-dev/pretrained-models/elmo/yahoo/model.tar.gz",
            "in-domain": "s3://suching-dev/pretrained-models/in-domain-elmo/yahoo/model.tar.gz"
        },
        "bert": {
            "weights": "s3://suching-dev/pretrained-models/bert/yahoo/model.tar.gz",
            "vocab": "s3://suching-dev/pretrained-models/bert/yahoo/vocab.txt"
        },
        "vae": {
            "model_archive": "s3://suching-dev/pretrained-models/vae_best_npmi/yahoo/model.tar.gz",
            "vocab": "s3://suching-dev/pretrained-models/vae_best_npmi/yahoo/vae.txt",
            "bg_freq": "s3://suching-dev/pretrained-models/vae_best_npmi/yahoo/vae.bgfreq.json"
        }
    },
    "hatespeech": {
        "train": "s3://suching-dev/final-datasets/hatespeech/train.jsonl",
        "dev": "s3://suching-dev/final-datasets/hatespeech/dev.jsonl",
        "test": "s3://suching-dev/final-datasets/hatespeech/test.jsonl",
        "unlabeled": "s3://suching-dev/final-datasets/hatespeech/unlabeled.jsonl",
        "reference_counts":  "s3://suching-dev/final-datasets/hatespeech/valid_npmi_reference/train.npz",
        "reference_vocabulary":  "s3://suching-dev/final-datasets/hatespeech/valid_npmi_reference/train.vocab.json",
        "stopword_path": "s3://suching-dev/stopwords/snowball_stopwords.txt",
        "glove": "s3://suching-dev/pretrained-models/glove/hatespeech/vectors.txt",
        "elmo": {
            "frozen": "s3://allennlp/models/transformer-elmo-2019.01.10.tar.gz",
            "fine-tuned": "s3://suching-dev/pretrained-models/elmo/hatespeech/model.tar.gz",
            "in-domain": "s3://suching-dev/pretrained-models/in-domain-elmo/hatespeech/model.tar.gz"
        },
        "bert": {
            "weights": "s3://suching-dev/pretrained-models/bert/hatespeech/model.tar.gz",
            "vocab": "s3://suching-dev/pretrained-models/bert/hatespeech/vocab.txt"
        },
        "vae": {
            "model_archive": "s3://suching-dev/pretrained-models/vae_best_npmi/hatespeech/model.tar.gz",
            "vocab": "s3://suching-dev/pretrained-models/vae_best_npmi/hatespeech/vae.txt",
            "bg_freq": "s3://suching-dev/pretrained-models/vae_best_npmi/hatespeech/vae.bgfreq.json"
        }
    },
    "ag-news": {
        "train": "s3://suching-dev/final-datasets/ag-news/train.jsonl",
        "dev": "s3://suching-dev/final-datasets/ag-news/dev.jsonl",
        "test": "s3://suching-dev/final-datasets/ag-news/test.jsonl",
        "unlabeled": "s3://suching-dev/final-datasets/ag-news/unlabeled.jsonl",
        "reference_counts":  "s3://suching-dev/final-datasets/ag-news/valid_npmi_reference/train.npz",
        "reference_vocabulary":  "s3://suching-dev/final-datasets/ag-news/valid_npmi_reference/train.vocab.json",
        "stopword_path": "s3://suching-dev/stopwords/snowball_stopwords.txt",
        "glove": "s3://suching-dev/pretrained-models/glove/ag-news/vectors.txt",
        "elmo": {
            "frozen": "s3://allennlp/models/transformer-elmo-2019.01.10.tar.gz",
            "fine-tuned": "s3://suching-dev/pretrained-models/elmo/ag-news/model.tar.gz",
            "in-domain": "s3://suching-dev/pretrained-models/in-domain-elmo/ag-news/model.tar.gz"
        },
        "bert": {
            "weights": "s3://suching-dev/pretrained-models/bert/ag-news/model.tar.gz",
            "vocab": "s3://suching-dev/pretrained-models/bert/ag-news/vocab.txt"
        },
        "vae": {
            "model_archive": "s3://suching-dev/pretrained-models/vae_best_npmi/ag-news/model.tar.gz",
            "vocab": "s3://suching-dev/pretrained-models/vae_best_npmi/ag-news/vae.txt",
            "bg_freq": "s3://suching-dev/pretrained-models/vae_best_npmi/ag-news/vae.bgfreq.json"
        }
    }
}
