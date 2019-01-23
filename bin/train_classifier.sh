#!/bin/bash

config=$1
override=$2

serialization_dir=model_logs/${config}
training_config=training_config/${config}.json

if [ "$override" == "override" ]; then
  echo "overriding $serialization_dir..."
  sudo rm -rf $serialization_dir;
fi

allennlp train \
    --include-package models.baselines.logistic_regression \
    --include-package dataset_readers.semisupervised_text_classification_json \
    --include-package common.allennlp_bridge \
    --include-package modules.token_embedders.vae_token_embedder \
    -s $serialization_dir \
    $training_config