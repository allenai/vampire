# vae
Exploring Variational Autoencoders for Representation Learning in NLP

## Dependencies:

* allennlp


## Commands:

### RNN VAE

```allennlp train --include-package modules.rnn_vae --include-package models.vae_classify_decoder --include-package dataset_readers.textcat -s ./model_logs/rnn_vae_clf_decoder ./training_config/rnn_vae.json```

### BOW VAE
```sudo allennlp train --include-package modules.bow_vae --include-package models.vae_classify_decoder --include-package dataset_readers.textcat -s ./model_logs/bow_vae_clf_decoder ./training_config/bow_vae.json```


```sudo allennlp train --include-package modules.bow_vae --include-package models.vae_classify_decoder --include-package dataset_readers.textcat -s ./model_logs/bow_vae_clf_decoder ./training_config/bow_vae.json```
