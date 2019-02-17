

local CUDA_DEVICE =
  if std.parseInt(std.extVar("NUM_GPU")) == 0 then
    -1
  else if std.parseInt(std.extVar("NUM_GPU")) > 1 then
    std.range(0, std.extVar("NUM_GPU") - 1)
  else if std.parseInt(std.extVar("NUM_GPU")) == 1 then
    0;

local BERT_FIELDS = {
  "bert_indexer": {
    
  },
  "bert_embedder": {
    
  }
};

local BASE_READER(THROTTLE, USE_SPACY_TOKENIZER) = {
  "lazy": false,
  "type": "semisupervised_text_classification_json",
  "tokenizer": {
    "word_splitter": if USE_SPACY_TOKENIZER == 1 then "spacy" else "just_spaces",
  },
  "token_indexers": {
      "bert": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased"
    }
  },
  "sequence_length": 400,
  "sample": THROTTLE,
};



{
  "dataset_reader": BASE_READER(std.extVar("THROTTLE"), std.parseInt(std.extVar("USE_SPACY_TOKENIZER"))),
  "validation_dataset_reader": BASE_READER(null, std.parseInt(std.extVar("USE_SPACY_TOKENIZER"))),
   "datasets_for_vocab_creation": ["train"],
   "train_data_path": std.extVar("TRAIN_PATH"),
   "validation_data_path": std.extVar("DEV_PATH"),
  "iterator": {
        "type": "basic",
        "batch_size": 1
    },
  "model": {
    "type": "text_classification_tune_pretrained",
    "text_field_embedder": {
        "allow_unmatched_keys": true,
        "embedder_to_indexer_map": {
            "bert": ["bert", "bert-offsets"],
        },
        "bert": {
            "type": "bert-pretrained",
            "pretrained_model": "bert-base-uncased",
            "requires_grad": true,
            "top_layer_only": false
            }
    },
    "dropout": 0.5,
  },

  "trainer": {
        "cuda_device": 0,
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "cut_frac": 0.1,
            "num_epochs": 5,
            "num_steps_per_epoch": 8327
        },
        "num_epochs": 5,
        "num_serialized_models_to_keep": 1,
        "optimizer": {
            "type": "bert_adam",
            "lr": 1e-05,
            "parameter_groups": [
                [
                    [
                        "bias",
                        "gamma",
                        "beta"
                    ],
                    {
                        "weight_decay_rate": 0
                    }
                ]
            ],
        },
        "should_log_learning_rate": true,
        "validation_metric": "+accuracy"
    }
}
