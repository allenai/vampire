# vae
Exploring Variational Autoencoders for Representation Learning in NLP

## Dependencies:

* allennlp (master)

## Installation

```pip install https://github.com/allenai/allennlp@master```

## Commands:

### Unsupervised RNN-VAE
```allennlp train --include-package modules.rnn_vae --include-package models.vae --include-package dataset_readers.textcat  -s ./model_logs/rnn_vae ./training_config/rnn_vae.json```

### Unsupervised BOW-VAE
```allennlp train --include-package modules.bow_vae --include-package models.vae --include-package dataset_readers.textcat  -s ./model_logs/bow_vae ./training_config/bow_vae.json```

### Supervised RNN-VAE
```allennlp train --include-package modules.rnn_vae --include-package models.vae_classifier --include-package dataset_readers.textcat  -s ./model_logs/rnn_vae_clf ./training_config/rnn_vae_clf.json```

### Supervised BOW-VAE
```allennlp train --include-package modules.bow_vae --include-package models.vae_classifier --include-package dataset_readers.textcat  -s ./model_logs/bow_vae_clf ./training_config/bow_vae_clf.json```