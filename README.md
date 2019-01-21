# vae

*Exploring Variational Autoencoders for Representation Learning in NLP*


## Installation

First install `allennlp`. The latest pip package of `allennlp` should work for most use-cases, but best to install the latest unreleased version of allennlp.

```
$ pip install https://github.com/allenai/allennlp@master
```

You also should install necessary dependencies:

```
$ pip install -r requirements.txt
```

## Download Data

Download your dataset of interest, and make sure it is made up of json files, where each line of each file corresponds to a separate instance. Each line must contain a `text` field and optionally a `label` field.

For imdb, you can use the `bin/download_imdb.py` script to get the data:

```
$ python -m bin.download_imdb --root-dir dump/imdb
```

## Generate Splits
Once you've downloaded your dataset to a directory, run `bin/generate_data.py`. The following command will hold out 5000 instances from the training data for the dev set, since a dev set is not originally provided. It will also randomly throttle the training into five samples, for testing semi-supervised learning under low-data regimes. 

```
$ mkdir datasets/
$ python -m bin.generate_data -d dump/imdb -o datasets/imdb -s 100 200 500 1000 10000 -x 5000
```

The output of `bin/generate_data.py` will include numbered directories corresponding to the subsamples, a `full` directory corresponding to the full training data, dev data, and test data, and an `unlabeled` directory corresponding to unlabeled data, if it exists.

If unlabeled data does not exist in the original corpus, you can sample the training data for unlabeled data:

```
$ python -m bin.generate_data -d dump/imdb -o datasets/imdb -s 100 200 500 1000 10000 -x 5000 -u 1000
```

Just make sure the sample sizes for the unlabeled data and/or dev data you choose does not exceed the total size of the training data!

## Preprocess Splits


Run `bin/preprocess.py` on the data directory containing your data splits:


```
$ python -m bin.preprocess --data_dir datasets/imdb/full/ --vocab_size 30000 --stopwords "snowball" --lower
$ python -m bin.preprocess --data_dir datasets/imdb/unlabeled/ --vocab_size 30000 --stopwords "snowball" --lower
$ python -m bin.preprocess --data_dir datasets/imdb/100/ --vocab_size 30000 --stopwords "snowball" --lower
$ python -m bin.preprocess --data_dir datasets/imdb/200/ --vocab_size 30000 --stopwords "snowball" --lower
$ python -m bin.preprocess --data_dir datasets/imdb/500/ --vocab_size 30000 --stopwords "snowball" --lower
$ python -m bin.preprocess --data_dir datasets/imdb/1000/ --vocab_size 30000 --stopwords "snowball" --lower
$ python -m bin.preprocess --data_dir datasets/imdb/10000/ --vocab_size 30000 --stopwords "snowball" --lower
```

Each of the commands above will create a number of files/directories in the corresponding directory:

* `train.jsonl` - preprocessed training data according to the flags supplied
* `test.jsonl` - if test data exists, preprocessed test data according to the flags supplied
* `dev.jsonl` - if dev data exists, preprocessed dev data according to the flags supplied
* `train.bgfreq.json` - background frequency counts for training data
* `vocabulary/` - AllenNLP vocabulary directory, generated from training data only
* `train.txt` - preprocessed training data text, for use in ELMo training
* `dev.txt` - preprocessed dev data text, for use in ELMo training
* `test.txt` - preprocessed test data text, for use in ELMo training

## Pre-train VAE
Open one of the training configs (e.g. `training_config/nvdm/nvdm_unsupervised_imdb.json`), and point the following fields to corresponding values:

* ``training_data_path``: ``$ROOT_PROJECT_DIR/datasets/imdb/full/train.jsonl``
* ``validation_data_path`` : ``$ROOT_PROJECT_DIR/datasets/imdb/full/dev.jsonl``
* ``background_data_path`` : `$ROOT_PROJECT_DIR/datasets/imdb/full/train.bgfreq.json`,

Then run:

```
$ allennlp train \
    --include-package models.nvdm \
    --include-package dataset_readers.text_classification_json \
    -s ./model_logs/nvdm \
    ./training_config/nvdm/nvdm_unsupervised_imdb.json
```

## Use Pre-train VAE with downstream classifier

Open one of the training configs in `training_config/baselines` (e.g. `training_config/baselines/logistic_regression_vae.json`), and point the following fields to corresponding values:


* ``training_data_path``: ``$ROOT_PROJECT_DIR/datasets/imdb/100/train.jsonl``
* ``validation_data_path`` : ``$ROOT_PROJECT_DIR/datasets/imdb/full/dev.jsonl``
* ``background_data_path`` : `$ROOT_PROJECT_DIR/datasets/imdb/100/train.bgfreq.json`
* ``supervised_vocab_file`` : `$ROOT_PROJECT_DIR/datasets/imdb/100/vocabulary/tokens.txt`
* ``vae_vocab_file`` : `$ROOT_PROJECT_DIR/model_logs/nvdm/vocabulary/vae.txt`
* ``label_file`` : `$ROOT_PROJECT_DIR/datasets/imdb/full/vocabulary/labels.txt`
* ``model_archive`` : `$ROOT_PROJECT_DIR/model_logs/nvdm/model.tar.gz`

Note that the ``training_data_path`` and ``background data_path`` are set to respective files in `$ROOT_PROJECT_DIR/datasets/imdb/100/`, one of the training data subsamples. The ``model_archive`` field is specified within the ``vae_token_embedder``.

Then run:

```
$ allennlp train \
    --include-package models.baselines.logistic_regression \
    --include-package dataset_readers.text_classification_json \
    --include-package common.allennlp_bridge \
    --include-package modules.token_embedders.vae_token_embedder \
    -s ./model_logs/baseline_lr \
    ./training_config/baselines/logistic_regression_vae.json
```

## Evaluate

```
allennlp evaluate \
    --include-package models.baselines.logistic_regression \
    --include-package dataset_readers.text_classification_json \
    --include-package common.allennlp_bridge \
    --include-package modules.token_embedders.vae_token_embedder \ 
    ./model_logs/baseline_lr/model.tar.gz  \
    ./datasets/imdb/full/test.jsonl
```

## Relevant literature

* http://bjlkeng.github.io/posts/semi-supervised-learning-with-variational-autoencoders/
* https://arxiv.org/abs/1312.6114
* https://arxiv.org/abs/1705.09296
* https://arxiv.org/abs/1808.10805
