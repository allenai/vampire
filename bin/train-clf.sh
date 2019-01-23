#!/bin/bash

ser_dir=$1
model=$2
vae=$3
override=$4

serialization_dir=model_logs/${ser_dir}
if [ "$vae" == "vae" ]; then
  training_config=training_config/baselines/${model}_vae.json
else;
  training_config=training_config/baselines/${model}.json
fi

training_config=training_config/baselines/${config}.json

if [ "$override" == "override" ]; then
  echo "overriding $serialization_dir..."
  sudo rm -rf $serialization_dir;
fi

allennlp train \
    --include-package models.baselines.${model} \
    --include-package data.dataset_readers.semisupervised_text_classification_json \
    --include-package common.allennlp_bridge \
    --include-package modules.token_embedders.vae_token_embedder \
    --include-package data.tokenizers.regex_and_stopword_filter \
    -s $serialization_dir \
    $training_config