#!/bin/bash

config=$1
override=$2
debug=$3

serialization_dir=model_logs/${config}
training_config=training_config/${config}.json

if [ "$override" == "override" ]; then
  echo "overriding $serialization_dir..."
  sudo rm -rf $serialization_dir;
fi


if [ "$debug" == "debug" ]; then
    echo "training in debug mode..."
    mv $training_config temp.json
    jq '.dataset_reader.debug=true | .validation_dataset_reader.debug=true' temp.json > $training_config
    rm temp.json
elif [ "$debug" == "prod" ]; then
    echo "training in prod mode..."
    mv $training_config temp.json
    jq '.dataset_reader.debug=false | .validation_dataset_reader.debug=false' temp.json > $training_config
    rm temp.json
else
    echo "invalid parameter $debug. Must be one of 'debug' or 'prod'."
    exit 127
fi



allennlp train \
    --include-package modules.onehot_embedder \
    --include-package models.nvdm \
    --include-package models.nvrnn \
    --include-package dataset_readers.textcat \
    -s $serialization_dir \
    $training_config
