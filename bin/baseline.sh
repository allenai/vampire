#!/bin/bash

sudo rm -rf /home/ubuntu/vae/model_logs/baseline
allennlp train --include-package models.baselines.seq2seq_classifier --include-package models.nvdm --include-package models.nvrnn --include-package dataset_readers.sst --include-package dataset_readers.textcat -s /home/ubuntu/vae/model_logs/baseline /home/ubuntu/vae/training_config/baselines/bilstm_maxpool_clf.json
# allennlp train --include-package models.baselines.logistic_regression --include-package dataset_readers.textcat -s /home/ubuntu/vae/model_logs/baseline /home/ubuntu/vae/training_config/baselines/logistic_regression_clf.json