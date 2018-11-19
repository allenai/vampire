# vae
Exploring Variational Autoencoders for Representation Learning in NLP

## Dependencies:

* allennlp


## Commands:

```allennlp make-vocab --include-package vae.allennlp_bridge --include-package vae.textcat -s $SERIALIZATION_DIR ./training_config/vocab.json```

```allennlp train --include-package vae.[bow_vae|rnn_vae] --include-package vae.allennlp_bridge --include-package vae.vae_classifier --include-package vae.textcat -s $SERIALIZATION_DIR ./training_config/vae.json```