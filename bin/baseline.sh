#!/bin/bash

sudo rm -rf /home/ubuntu/vae/model_logs/baseline
allennlp train --include-package models.baselines.seq2seq_classifier --include-package modules.scholar --include-package models.semisupervised_vae --include-package dataset_readers.textcat -s /home/ubuntu/vae/model_logs/baseline /home/ubuntu/vae/training_config/baselines/bilstm_maxpool_clf.json