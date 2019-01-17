# vae

*Exploring Variational Autoencoders for Representation Learning in NLP*


## Installation

First install `allennlp`. The latest pip package of `allennlp` should work for most use-cases, but best to install the latest unreleased version of allennlp.

```
$ pip install https://github.com/allenai/allennlp@master
```

## Setup

Download your dataset of interest, and make sure it's json files of format `{"text": ..., "label": ...}`.

Run `bin/preprocess_data.py` on the data directory containing your files.

Open one of ``scholar.json`` or ``nvdm_unsupervised.json`` and point the ``training_data_path`` and ``validation_data_path`` to your files.


## Run

```
$ allennlp train \
        --include-package models.nvdm \
        --include-package dataset_readers.text_classification_json \
        -s ./model_logs/nvdm \
        ./training_config/nvdm/nvdm_unsupervised.json
```

## Relevant literature

* http://bjlkeng.github.io/posts/semi-supervised-learning-with-variational-autoencoders/
* https://arxiv.org/abs/1312.6114
* https://arxiv.org/abs/1705.09296
* https://arxiv.org/abs/1808.10805
