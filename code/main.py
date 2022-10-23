"""
A wrapper script for debug and multi-gpu.
"""

import argparse
import json
import os

_ARG_PARSER = argparse.ArgumentParser(description="我的实验，需要指定配置文件")
_ARG_PARSER.add_argument('--cuda', '-c', type=str, default='0', help='gpu ids, like: 1,2,3')
_ARG_PARSER.add_argument('--seed', '-s', type=str, default='42', help='random seed.')
_ARG_PARSER.add_argument('--name', '-n', type=str, default='debug', help='save name.')
_ARG_PARSER.add_argument('--config', type=str, default='mdetr', help='configuration file name.')

_ARG_PARSER.add_argument(
    "--node-rank", type=int, default=0, help="rank of this node in the distributed setup"
)
_ARG_PARSER.add_argument(
    "--file-friendly-logging",
    action="store_true",
    default=True,
    help="outputs tqdm status on separate lines and slows tqdm refresh rate",
)
_ARG_PARSER.add_argument(
    "-f",
    "--force",
    action="store_true",
    default=False,
    help="overwrite the output directory if it exists",
)

_ARG_PARSER.add_argument('--batch-size', type=int, default=16, help='batch size')
_ARG_PARSER.add_argument('--grad_norm', type=float, default=0.1)
_ARG_PARSER.add_argument('--lr', type=float, default=1e-5)
_ARG_PARSER.add_argument('--head_lr', type=float, default=1e-4)
_ARG_PARSER.add_argument('--gnn_lr', type=float, default=1e-4)
_ARG_PARSER.add_argument('--graph_layers', type=int, default=0, help='graph layers')

_ARG_PARSER.add_argument('--data_sufix', type=str, default='', help='data sufix')
_ARG_PARSER.add_argument("--no_cluster_edge", action="store_true", default=False)
_ARG_PARSER.add_argument("--no_span_edge", action="store_true", default=False)
_ARG_PARSER.add_argument("--only_nouns", action="store_true", default=False)
_ARG_PARSER.add_argument("--add_gold_spans", action="store_true", default=False)

_ARGS = _ARG_PARSER.parse_args()

if len(_ARGS.cuda) > 1:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

os.environ["OMP_NUM_THREADS"] = '8'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # if cuda >= 10.2
os.environ['CUDA_VISIBLE_DEVICES'] = _ARGS.cuda

os.environ["EXP_ARGS"] = json.dumps(dict(
    seed=_ARGS.seed, batch_size=_ARGS.batch_size, grad_norm=_ARGS.grad_norm,
    lr=_ARGS.lr, head_lr=_ARGS.head_lr, gnn_lr=_ARGS.gnn_lr,
    graph_layers=_ARGS.graph_layers, data_sufix=_ARGS.data_sufix,
    no_cluster_edge=_ARGS.no_cluster_edge, no_span_edge=_ARGS.no_span_edge,
    only_nouns=_ARGS.only_nouns, add_gold_spans=_ARGS.add_gold_spans
))


def main(args: argparse.Namespace):
    import torch
    from allennlp.common.params import Params
    from allennlp.commands.train import train_model
    import allennlp_models  # noqa: F401
    import src  # noqa: F401

    # cuDNN deterministically select an algorithm
    torch.backends.cudnn.benchmark = False
    # make cuDNN selected algorithms deterministic
    torch.backends.cudnn.deterministic = True
    # make other PyTorch operations behave deterministically
    # torch.use_deterministic_algorithms(True)  # not support pyg

    params = Params.from_file(f"config/{args.config}.jsonnet")

    # To support multi-gpu in this script, we mannually start the training.
    train_model(
        params,
        f"results/{_ARGS.name}-{_ARGS.seed}",
        force=args.force,
        node_rank=args.node_rank,
        # dry_run=True,
        file_friendly_logging=args.file_friendly_logging
    )


if __name__ == "__main__":
    main(_ARGS)
