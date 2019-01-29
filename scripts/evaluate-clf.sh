#!/bin/bash

model=$2
model_archive=$2
test_file=$3

allennlp evaluate \
    --include-package vae.models.baselines.${model} \
    --include-package vae.data.dataset_readers.semisupervised_text_classification_json \
    --include-package vae.common.allennlp_bridge \
    --include-package vae.modules.token_embedders.vae_token_embedder \
    --include-package vae.data.tokenizers.regex_and_stopword_filter \
    -s $model_archive \
    $test_file