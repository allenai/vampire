mkdir -p $(pwd)/examples/ag
curl -Lo $(pwd)/examples/ag/train.jsonl https://s3-us-west-2.amazonaws.com/allennlp/datasets/ag-news/train.jsonl
curl -Lo $(pwd)/examples/ag/dev.jsonl https://s3-us-west-2.amazonaws.com/allennlp/datasets/ag-news/dev.jsonl
curl -Lo $(pwd)/examples/ag/test.jsonl https://s3-us-west-2.amazonaws.com/allennlp/datasets/ag-news/test.jsonl