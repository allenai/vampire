#!/bin/bash

clf=$1
ser_dir=$2
config=$3
override=$4

serialization_dir=${ser_dir}
training_config=$config

if [ "$override" == "override" ]; then
  echo "overriding $serialization_dir..."
  sudo rm -rf $serialization_dir;
fi

allennlp train \
    --include-package vae.models.baselines.${clf} \
    --include-package vae.data.dataset_readers.semisupervised_text_classification_json \
    --include-package vae.common.allennlp_bridge \
    --include-package vae.modules.token_embedders.vae_token_embedder \
    --include-package vae.data.tokenizers.regex_and_stopword_filter \
    -s $serialization_dir \
    $training_config