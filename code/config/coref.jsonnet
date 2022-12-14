// Configuration for a coreference resolution model based on:
//   Lee, Kenton et al. “Higher-order Coreference Resolution with Coarse-to-fine Inference.” NAACL (2018).
//   + SpanBERT-large

local cuda = std.extVar("CUDA_VISIBLE_DEVICES");
// local max_length = 384;  // max words: 323, max word pieces: 371
local cross_no = 0;

local transformer_model = "/home/ubuntu/models/spanbert-large-cased";
local max_length = 512;
local feature_size = 20;
local max_span_width = 30;

local transformer_dim = 1024;  # uniquely determined by transformer_model
local span_embedding_dim = 3 * transformer_dim + feature_size;
local span_pair_embedding_dim = 3 * span_embedding_dim + feature_size;

{
  "dataset_reader": {
    "type": "coref_my",
    cross_no: cross_no,
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer_mismatched",
        "model_name": transformer_model,
        "max_length": max_length
      },
    },
    "max_span_width": max_span_width,
  },
  "train_data_path": "data/res3.jsonlines",
  "validation_data_path": self.train_data_path + ".dev",
  "model": {
    "type": "coref",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "pretrained_transformer_mismatched",
            "model_name": transformer_model,
            "max_length": max_length
        }
      }
    },
    "context_layer": {
        "type": "pass_through",
        "input_dim": transformer_dim
    },
    "mention_feedforward": {
        "input_dim": span_embedding_dim,
        "num_layers": 2,
        "hidden_dims": 1500,
        "activations": "relu",
        "dropout": 0.3
    },
    "antecedent_feedforward": {
        "input_dim": span_pair_embedding_dim,
        "num_layers": 2,
        "hidden_dims": 1500,
        "activations": "relu",
        "dropout": 0.3
    },
    "initializer": {
      "regexes": [
        [".*_span_updating_gated_sum.*weight", {"type": "xavier_normal"}],
        [".*linear_layers.*weight", {"type": "xavier_normal"}],
        [".*scorer.*weight", {"type": "xavier_normal"}],
        ["_distance_embedding.weight", {"type": "xavier_normal"}],
        ["_span_width_embedding.weight", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_ih.*", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_hh.*", {"type": "orthogonal"}]
      ]
    },
    "feature_size": feature_size,
    "max_span_width": max_span_width,
    "spans_per_word": 0.4,
    "max_antecedents": 50,
    "coarse_to_fine": true,
    "inference_order": 2
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      # Explicitly specifying sorting keys since the guessing heuristic could get it wrong
      # as we a span field.
      "sorting_keys": ["text"],
      "batch_size": 4
    }
  },
  "trainer": {
    [if std.length(cuda) < 2 then "cuda_device"]: if std.length(cuda) > 0 then 0 else -1,
    "num_epochs": 40,
    "patience" : 10,
    "validation_metric": "+coref_f1",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 3e-4,
      "parameter_groups": [
        [[".*transformer.*"], {"lr": 1e-5}]
      ]
    }
  }
}