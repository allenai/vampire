# vae
Exploring Variational Autoencoders for Representation Learning in NLP

## Dependencies:

* allennlp


## Commands:

```allennlp make-vocab --include-package common.allennlp_bridge --include-package dataset_readers.textcat -s $SERIALIZATION_DIR ./training_config/vocab.json```

```allennlp train --include-package modules.[bow_vae|rnn_vae] --include-package common.allennlp_bridge --include-package models.vae_classifier --include-package dataset_readers.textcat -s $SERIALIZATION_DIR ./training_config/vae.json```