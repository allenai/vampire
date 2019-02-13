local CUDA_DEVICE =
  if std.parseInt(std.extVar("NUM_GPU")) == 0 then
    -1
  else if std.parseInt(std.extVar("NUM_GPU")) > 1 then
    std.range(0, std.extVar("NUM_GPU") - 1)
  else if std.parseInt(std.extVar("NUM_GPU")) == 1 then
    0;

// Add VAE embeddings to the input of the classifier.
local ADD_VAE = 0;
local VAE_MODEL = "s3://best_nll_vae/model.tar.gz";
local VAE_BG_FREQ = "s3://best_nll_vae/vae.bgfreq.json";
local VAE_VOCAB = "s3://best_nll_vae/vae.txt";

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
                "model_archive": VAE_MODEL,
                "background_frequency": VAE_BG_FREQ,
                "dropout": 0.2
        }
    }
};

local VOCABULARY_WITH_VAE = {
  "vocabulary":{
              "type": "vocabulary_with_vae",
              "vae_vocab_file": VAE_VOCAB,
          }
};


{
    "numpy_seed": 20203,
    "pytorch_seed": 20203,
    "random_seed": 20203,
    "dataset_reader": {
        "type": "semisupervised_text_classification_json",
        "lazy": false,
        "sample": "200",
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
        "dropout": 0.2,
        "encoder": {
            "type": "seq2vec",
            "architecture": {
                "type": "cnn",
                "embedding_dim": 512,
                "ngram_filter_sizes": [
                    1,
                    2,
                    3,
                    4,
                    5
                ],
                "num_filters": 512,
                "output_dim": 64
            }
        },
        "input_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 512,
                    "trainable": true,
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
            "lr": 0.0005
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
            }  + if ADD_VAE == 1 then VAE_FIELDS['vae_indexer'] else {},
        },
        "tokenizer": {
            "word_splitter": "spacy"
        }
    }
} + if ADD_VAE == 1 then VOCABULARY_WITH_VAE else {}