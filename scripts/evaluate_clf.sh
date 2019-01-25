#!/bin/bash

ser_dir=$1
model=$2
model_file=$2
test_file=$3

model_archive=model_logs/${model}
model_archive=model_logs/${model_file}

training_config=training_config/baselines/${config}.json

allennlp evaluate \
    --include-package models.baselines.${model} \
    --include-package data.dataset_readers.semisupervised_text_classification_json \
    --include-package common.allennlp_bridge \
    --include-package modules.token_embedders.vae_token_embedder \
    --include-package data.tokenizers.regex_and_stopword_filter \
    -s $model_archive \
    $test_file