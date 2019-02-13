[![codecov](https://codecov.io/gh/allenai/vae/branch/master/graph/badge.svg?token=NOriF2Rm8p)](https://codecov.io/gh/allenai/vae)

# vae

*Exploring Variational Autoencoders for Representation Learning in NLP*


## Installation

Install necessary dependencies via `requirements.txt`, which will include the latest unreleased install of `allennlp` (from the `master` branch).

```
$ pip install -r requirements.txt
```

Verify your installation by running: 

```
$ pytest -v --color=yes vae
```

All tests should pass.

## Download Data

Download your dataset of interest, and make sure it is made up of json files, where each line of each file corresponds to a separate instance. Each line must contain a `text` field, and optionally a `label` field.

For imdb, you can use the `bin/download_imdb.py` script to get the data:

```
$ python -m bin.download_imdb --dest dump/imdb
```

## Generate Splits
Once you've downloaded your dataset to a directory, run `bin/generate_data.py` if you'd like to split the training data into development data or unlabeled data. The script will output your files to a specified output directory. The following command will hold out 5000 instances from the training data for the dev set:

```
$ mkdir datasets/
$ python -m bin.generate_data -d dump/imdb -o datasets/imdb -x 5000
```

If unlabeled data does not exist in the original corpus, you can sample the training data for unlabeled data:

```
$ python -m bin.generate_data -d dump/imdb -o datasets/imdb -x 5000 -u 1000
```

Just make sure the sample sizes for the unlabeled data and/or dev data you choose does not exceed the total size of the training data!


## Pre-train a VAE

```
$ NUM_GPU=0 python -m scripts.train -x 1110101010 -c ./training_config/local/vae_unsupervised.jsonnet -s ./model_logs/vae_unsupervised --override
```

This command will output model_logs at `./model_logs/vae_unsupervised` from the training config `./training_config/nvdm/vae_unsupervised.jsonnet`. The `override` flag will override previous experiment at the same serialization directory.

## Use Pre-train VAE with downstream classifier

```
$ NUM_GPU=0 python -m scripts.train -x 1110101010 -c ./training_config/local/lr_classifier.jsonnet -s ./model_logs/lr_vae --override
```

This command will output model_logs at `./model_logs/lr_vae` from the training config `./training_config/local/classifier.jsonnet`. The `override` flag will override previous experiment at the same serialization directory.

You can change the `VAE_FIELDS` in the `lr_classifier.jsonnet` to your newly trained VAE:

```
local VAE_FIELDS = {
    "vae_indexer": {
        "vae_tokens": {
            "type": "single_id",
            "namespace": "vae",
            "lowercase_tokens": true
        }
    },  
    "vae_embedder": {
        "vae_tokens": {
                "type": "vae_token_embedder",
                "representation": "encoder_output",
                "expand_dim": true,
                "model_archive": "/path/to/model_logs/vae_unsupervised/model.tar.gz",
                "background_frequency": "/path/to/model_logs/vae_unsupervised/vocabulary/vae.bgfreq.json",
                "dropout": 0.2
        }
    }
};
```

*Note* : You can additionally subsample the training data by setting `{"dataset_reader": {"sample": N}}` where `N < len(train.jsonl)`.

## Evaluate

```
$ ./bin/evaluate-clf.sh logistic_regression ./model_logs/lr_vae/model.tar.gz ./datasets/imdb/test.jsonl
```

## Relevant literature

* http://bjlkeng.github.io/posts/semi-supervised-learning-with-variational-autoencoders/
* https://arxiv.org/abs/1312.6114
* https://arxiv.org/abs/1705.09296
* https://arxiv.org/abs/1808.10805
