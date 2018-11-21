# vae

*Exploring Variational Autoencoders for Representation Learning in NLP*


## Installation

First install `allennlp`:

```
$ pip install https://github.com/allenai/allennlp@master
```

Then run one of the following commands:

*Unsupervised RNN-VAE*
```
$ allennlp train --include-package modules.rnn_vae --include-package models.vae --include-package dataset_readers.textcat  -s ./model_logs/rnn_vae ./training_config/rnn_vae.json
```

*Unsupervised BOW-VAE*
```
$ allennlp train --include-package modules.bow_vae --include-package models.vae --include-package dataset_readers.textcat  -s ./model_logs/bow_vae ./training_config/bow_vae.json
```

*Supervised RNN-VAE*
```
$ allennlp train --include-package modules.rnn_vae --include-package models.vae_classifier --include-package dataset_readers.textcat  -s ./model_logs/rnn_vae_clf ./training_config/rnn_vae_clf.json
```

*Supervised BOW-VAE*
```
$ allennlp train --include-package modules.bow_vae --include-package models.vae_classifier --include-package dataset_readers.textcat  -s ./model_logs/bow_vae_clf ./training_config/bow_vae_clf.json
```

## Relevant literature

* http://bjlkeng.github.io/posts/semi-supervised-learning-with-variational-autoencoders/
* https://arxiv.org/abs/1312.6114
* https://arxiv.org/abs/1705.09296
* https://arxiv.org/abs/1808.10805
