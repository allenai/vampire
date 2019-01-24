#!/bin/bash

ser_dir=$1
config=$2
clf=$3
override=$4

serialization_dir=${ser_dir}
training_config=$config

if [ "$override" == "override" ]; then
  echo "overriding $serialization_dir..."
  sudo rm -rf $serialization_dir;
fi

allennlp train \
    --include-package models.baselines.${clf} \
    --include-package data.dataset_readers.semisupervised_text_classification_json \
    --include-package common.allennlp_bridge \
    --include-package modules.token_embedders.vae_token_embedder \
    --include-package data.tokenizers.regex_and_stopword_filter \
    -s $serialization_dir \
    $training_config