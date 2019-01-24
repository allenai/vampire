#!/bin/bash

ser_dir=$1
config=$2
override=$3

serialization_dir=$ser_dir
training_config=$config

if [ "$override" == "override" ]; then
  echo "overriding $serialization_dir..."
  sudo rm -rf $serialization_dir;
fi

allennlp train \
  --include-package models.nvdm \
  --include-package data.dataset_readers.semisupervised_text_classification_json \
  --include-package data.tokenizers.regex_and_stopword_filter \
  --include-package common.allennlp_bridge \
  -s $serialization_dir \
  $training_config