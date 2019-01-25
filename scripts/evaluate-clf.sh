#!/bin/bash

model=$2
model_archive=$2
test_file=$3

allennlp evaluate \
    --include-package models.baselines.${model} \
    --include-package data.dataset_readers.semisupervised_text_classification_json \
    --include-package common.allennlp_bridge \
    --include-package modules.token_embedders.vae_token_embedder \
    --include-package data.tokenizers.regex_and_stopword_filter \
    -s $model_archive \
    $test_file