mkdir -p $(pwd)/examples/imdb
curl -Lo $(pwd)/examples/imdb/train.jsonl https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/train.jsonl
curl -Lo $(pwd)/examples/imdb/dev.jsonl https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/dev.jsonl
curl -Lo $(pwd)/examples/imdb/test.jsonl https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/test.jsonl