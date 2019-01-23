#!/bin/bash

ser_dir=$1
config=$2
override=$3

serialization_dir=model_logs/${ser_dir}
training_config=training_config/${config}.json

if [ "$override" == "override" ]; then
  echo "overriding $serialization_dir..."
  sudo rm -rf $serialization_dir;
fi

allennlp train \
    --include-package models.baselines.logistic_regression \
    --include-package data.dataset_readers.semisupervised_text_classification_json \
    --include-package common.allennlp_bridge \
    --include-package modules.token_embedders.vae_token_embedder \
    --include-package models.nvdm \
    --include-package data.tokenizers.regex_and_stopword_filter \
    -s $serialization_dir \
    $training_config