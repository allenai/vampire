#!/bin/bash

rm -rf debug

serialization_dir=$1
training_config=$2
override=$3

if [ "$override" == "override" ]; then
  echo "overriding $serialization_dir..."
  sudo rm -rf $serialization_dir;
fi

allennlp train \
  --include-package vae.models.unsupervised \
  --include-package vae.data.dataset_readers.semisupervised_text_classification_json \
  --include-package vae.data.tokenizers.regex_and_stopword_filter \
  --include-package vae.common.allennlp_bridge \
  -s $serialization_dir \
  $training_config

