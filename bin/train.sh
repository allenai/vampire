#!/bin/bash

mode=$1
arch=$2
dataset=$3
serialization_dir=$4
override=$5
debug=$6

if [ "$5" == "override" ]; then
  echo "overriding $serialization_dir..."
  sudo rm -rf $serialization_dir;
fi

if [ "$1" == "m1" ]; then
    model_mode="unsupervised_vae"
elif [ "$1" == "m2" ]; then
    model_mode="semisupervised_vae"
elif [ "$1" == "scholar" ]; then
    model_mode="semisupervised_vae"
elif [ "$1" == "nvdm" ]; then
    model_mode="nvdm"
elif [ "$1" == "nvrnn" ]; then
    model_mode="nvrnn"
elif [ "$1" == "rnnlm" ]; then
    model_mode="rnnlm"
else
    echo "invalid mode $mode. Must be one of 'm1', 'm2', 'scholar', 'nvdm', 'nvrnn', or 'rnnlm'."
    exit 127
fi

if [ "$6" == "debug" ]; then
    echo "training in debug mode..."
    mv ./training_config/$mode/${arch}_${dataset}.json temp.json
    jq '.dataset_reader.debug=true | .validation_dataset_reader.debug=true' temp.json > ./training_config/$mode/${arch}_${dataset}.json
    rm temp.json
elif [ "$6" == "prod" ]; then
    echo "training in prod mode..."
    mv ./training_config/$mode/${arch}_${dataset}.json temp.json
    jq '.dataset_reader.debug=false | .validation_dataset_reader.debug=false' temp.json > ./training_config/$mode/${arch}_${dataset}.json
    rm temp.json
else
    echo "invalid parameter $debug. Must be one of 'debug' or 'prod'."
    exit 127
fi



allennlp train --include-package modules.onehot_embedder --include-package models.${model_mode} --include-package dataset_readers.textcat -s $serialization_dir ./training_config/$mode/${arch}_${dataset}.json