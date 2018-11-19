# vae
Exploring Variational Autoencoders for Representation Learning in NLP

## Dependencies:

* allennlp


## Commands:

```allennlp make-vocab --include-package common.allennlp_bridge --include-package dataset_readers.vocab_generator -s $SERIALIZATION_DIR ./training_config/vocab.json```

### RNN VAE

```allennlp train --include-package modules.rnn_vae --include-package common.allennlp_bridge --include-package models.vae_classifier --include-package dataset_readers.textcat -s ./model_logs/rnn_vae ./training_config/rnn_vae.json```

### BOW VAE
```allennlp train --include-package modules.bow_vae --include-package common.allennlp_bridge --include-package models.vae_classifier --include-package dataset_readers.textcat -s ./model_logs/bow_vae ./training_config/bow_vae.json```