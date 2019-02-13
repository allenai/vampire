local CUDA_DEVICE =
  if std.parseInt(std.extVar("NUM_GPU")) == 0 then
    -1
  else if std.parseInt(std.extVar("NUM_GPU")) > 1 then
    std.range(0, std.extVar("NUM_GPU") - 1)
  else if std.parseInt(std.extVar("NUM_GPU")) == 1 then
    0;


// Add VAE embeddings to the input of the classifier.
local ADD_VAE = 1;


local VAE_FIELDS = {
    "vae_indexer": {
        "vae_tokens": {
            "type": "single_id",
            "namespace": "vae",
            "lowercase_tokens": true
        }
    },  
    "vae_embedder": {
        "vae_tokens": {
                "type": "vae_token_embedder",
                "representations": ["first_layer_output"],
                "expand_dim": false,
                "model_archive": "s3://best-vae/model.tar.gz",
                "background_frequency": "s3://best-vae/vae.bgfreq.json",
                "dropout": 0.0
        }
    }
};

local VOCABULARY_WITH_VAE = {
  "vocabulary":{
              "type": "vocabulary_with_vae",
              "vae_vocab_file": "s3://best-vae/vae.txt",
          }
};


{
    "numpy_seed": 16011988,
    "random_seed": 16011988,
    "pytorch_seed": 16011988,
    "dataset_reader": {
        "type": "semisupervised_text_classification_json",
        "lazy": false,
        "sample": null,
        "sequence_length": 400,
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true,
                "namespace": "classifier"
            } 
        } + if ADD_VAE == 1 then VAE_FIELDS['vae_indexer'] else {},
        "tokenizer": {
            "word_splitter": "spacy"
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size": 128
    },
    "model": {
        "type": "classifier",
        "dropout": 0.5,
        "input_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "bag_of_word_counts",
                    "ignore_oov": "true",
                    "vocab_namespace": "classifier"
                }
            } + if ADD_VAE == 1 then VAE_FIELDS['vae_embedder'] else {}
        }
    },
    "train_data_path": "s3://suching-dev/imdb/train.jsonl",
    "validation_data_path": "s3://suching-dev/imdb/dev.jsonl",
    "trainer": {
        "cuda_device": CUDA_DEVICE,
        "num_epochs": 200,
        "optimizer": {
            "type": "adam",
            "lr": 0.0001
        },
        "patience": 20,
        "validation_metric": "+accuracy"
    },
    "datasets_for_vocab_creation": [
        "train"
    ],
    "validation_dataset_reader": {
        "type": "semisupervised_text_classification_json",
        "lazy": false,
        "sample": null,
        "sequence_length": 400,
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true,
                "namespace": "classifier"
            } 
        } + if ADD_VAE == 1 then VAE_FIELDS['vae_indexer'] else {},
        "tokenizer": {
            "word_splitter": "spacy"
        }
    }
} + if ADD_VAE == 1 then VOCABULARY_WITH_VAE else {}