

local CUDA_DEVICE =
  if std.parseInt(std.extVar("NUM_GPU")) == 0 then
    -1
  else if std.parseInt(std.extVar("NUM_GPU")) > 1 then
    std.range(0, std.extVar("NUM_GPU") - 1)
  else if std.parseInt(std.extVar("NUM_GPU")) == 1 then
    0;


local BASE_READER(THROTTLE, USE_SPACY_TOKENIZER) = {
  "lazy": false,
  "type": "semisupervised_text_classification_json",
  "tokenizer": {
    "word_splitter": if USE_SPACY_TOKENIZER == 1 then "spacy" else "just_spaces",
  },
  "token_indexers": {
      "elmo": {
        "type": "elmo_characters"
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
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size": std.parseInt(std.extVar("BATCH_SIZE")),
    "biggest_batch_first": false,
  },
  "model": {
    "type": "text_classification_tune_pretrained",
    "text_field_embedder": {
        "elmo":{
            "type": "elmo_token_embedder",
            "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
            "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
            "do_layer_norm": false,
            "dropout": 0.0,
            "scalar_mix_parameters": [0.0, 0.0, 20.0],
            "requires_grad": true,
        },
    },
    "dropout": 0.5,
  },

  "trainer": {
      "optimizer": {
            "type": "adam",
            "lr": 0.0005,
            "parameter_groups": [
                  [["text_field_embedder.token_embedder_elmo._elmo._elmo_lstm._token_embedder.*"], {}],
                  [["text_field_embedder.token_embedder_elmo._elmo._elmo_lstm._elmo_lstm.forward_layer_0.*", "text_field_embedder.token_embedder_elmo._elmo._elmo_lstm._elmo_lstm.backward_layer_0.*"], {}],
                  [["text_field_embedder.token_embedder_elmo._elmo._elmo_lstm._elmo_lstm.forward_layer_1.*", "text_field_embedder.token_embedder_elmo._elmo._elmo_lstm._elmo_lstm.backward_layer_1.*"], {}],
                  [["^output_layer.weight", "^output_layer.bias", ".*scalar_mix.*"], {}]
            ],
        },

    "validation_metric": "+accuracy",
    "should_log_learning_rate": true,

    "num_epochs": 15,
    "cuda_device": CUDA_DEVICE,
    "num_serialized_models_to_keep": 1,
    "grad_norm": 5.0,

    "learning_rate_scheduler": {
        "type": "slanted_triangular",
        "gradual_unfreezing": true,
        "discriminative_fine_tuning": true,
        "num_epochs": 15,
        "ratio": 32,
        "decay_factor": 0.4,
        // 98794 training instances for use-trees and sst-2
        "num_steps_per_epoch": 988,
    }
  }

}
