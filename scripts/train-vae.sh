#!/bin/bash

ser_dir=$1
config=$2
override=$3

serialization_dir=$ser_dir
training_config=$config

if [ "$override" == "override" ]; then
  echo "overriding $serialization_dir..."
  rm -rf $serialization_dir;
fi

allennlp train \
  --include-package vae.models.unsupervised \
  --include-package vae.models.joint_semi_supervised \
  --include-package vae.data.dataset_readers.semisupervised_text_classification_json \
  --include-package vae.data.tokenizers.regex_and_stopword_filter \
  --include-package vae.common.allennlp_bridge \
  -s $serialization_dir \
  $training_config