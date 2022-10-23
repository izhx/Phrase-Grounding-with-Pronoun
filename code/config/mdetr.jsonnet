local cuda = std.extVar("CUDA_VISIBLE_DEVICES");

local mdetr_file = "/home/ubuntu/models/pretrained_EB3_checkpoint.pth";

local max_length = 383;  // max words: 323, max word pieces: 371

local args = std.parseJson(std.extVar("EXP_ARGS"));

{
    dataset_reader : {
        type: "coref_ground_mdetr",
        image_dir: "data/images/",
        only_nouns: args.only_nouns,
        build_graph: if args.graph_layers > 0 then true else false,
        no_span_edge: args.no_span_edge,
        one_word_span: false,
        no_cluster_edge: args.no_cluster_edge,
        pseudo_cluster: if args.data_sufix == "" then false else true,
        add_gold_spans: args.add_gold_spans,
        num_graph_relations: 2 + (if self.no_span_edge then 0 else 2) + if self.no_cluster_edge then 0 else 1,
        timm_config: {
            input_size: [3, 300, 300], pool_size: [10, 10], crop_pct: 0.904
        },
        max_length: max_length,
        token_indexers: {
            roberta: {
                type: "pretrained_transformer_mismatched",
                model_name: "roberta-base",
                max_length: max_length
            },
        },
    },
    train_data_path: "data/train.json" + args.data_sufix,
    validation_data_path: "data/val.json" + args.data_sufix,
    test_data_path: "data/test.json" + args.data_sufix,
    evaluate_on_test: if self.test_data_path == null then false else true,
    data_loader: {
        batch_sampler: {
            type: "bucket",
            batch_size: args.batch_size,
            sorting_keys: ["text"]
        }
    },
    random_seed: args.seed,
    numpy_seed: self.random_seed,
    pytorch_seed: self.random_seed,
    model: {
        type: "phrase_grounding_wrapper",
        backbone: {
            type: "phrase_grounding_mdetr",
            mdetr_file: mdetr_file,
            // num_queries: 30,
            max_length: max_length,
            sub_token_mode: "first",
            [if args.graph_layers > 0 then "gnn"]: {
                in_channels: 768,
                hidden_channels: self.in_channels,
                num_layers: args.graph_layers,
                num_relations: $["dataset_reader"].num_graph_relations,
                pyg_kwargs: { type: "rgcn" }
            }
        }
    },
    trainer: {
        [if std.length(cuda) < 2 then "cuda_device"]: if std.length(cuda) > 0 then 0 else -1,
        num_epochs: 50,
        grad_norm: args.grad_norm,
        patience: 10,
        validation_metric: "+recall_1",
        optimizer: {
            type: "adamw",
            lr: args.lr,
            weight_decay: 1e-4,
            parameter_groups: [
                [[".*mdetr.class_embed.*"], {"lr": args.head_lr}],
                [[".*gnn.*"], {"lr": args.gnn_lr}],
            ]
        }
    }
}
