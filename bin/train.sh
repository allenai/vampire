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
    --include-package models.nvdm \
    --include-package models.nvrnn \
    --include-package dataset_readers.text_classification_json \
    --include-package common.allennlp_bridge \
    -s $serialization_dir \
    $training_config
