import copy
import gc
import json
import os
import sys

import torch
import argparse

from Exp import Exp


parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='PROTEINS')
parser.add_argument('--data_root', type=str, default='data/')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=80)
parser.add_argument('--num_workers', type=int, default=0,
                    help='DataLoader worker processes. 0 (default): collate in-process and '
                         'use --loader_prefetch to overlap the next batch with GPU work. '
                         '>0 uses spawn (CUDA-safe) plus pin_memory; can be slower for '
                         'in-memory TU datasets because whole batches are serialized')
parser.add_argument('--pin_memory', type=int, default=1, choices=[0, 1],
                    help='1 (default): pin collated CPU batches for async H2D when CUDA is used')
parser.add_argument('--loader_prefetch', type=int, default=2,
                    help='CPU batches prepared ahead of the train/eval loop when num_workers=0; '
                         '0 disables. Also used as DataLoader prefetch_factor when num_workers>0')
parser.add_argument('--persistent_workers', type=int, default=1, choices=[0, 1],
                    help='keep DataLoader workers alive across epochs when num_workers>0')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.5,
                    help='cosine annealing min lr ratio: eta_min = lr * lr_decay; 1.0 disables decay')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='L2 regularization coefficient in Adam (not learning rate decay)')
parser.add_argument('--dim_embedding', type=int, default=64)
parser.add_argument('--dim_embedding_gnn', type=int, default=128)
parser.add_argument('--num_edges2', type=int, default=64,
                    help='number of learned hyperedges in View T')
parser.add_argument('--num_edges1', type=int, default=None,
                    help='number of handcrafted hyperedges in View S; defaults to num_edges2 unless set via CLI or config')
parser.add_argument('--num_layers_gnn', type=int, default=3, help='number of layers in the first encoder embedding.') # 2
parser.add_argument('--num_anchors', type=int, default=64, help='number of anchor graphs to be used')
parser.add_argument('--weight_consistency', type=float, default=1,
                    help='weight of the two-view consistency loss')
parser.add_argument('--weight_hse', type=float, default=0.001, help="weight of hierarchical se loss")
parser.add_argument('--warm_epochs', type=int, default=2) 
parser.add_argument('--epoch_select', type=str, default='val_acc',
                    choices=['val_acc', 'val_acc_eq', 'val_loss_sup', 'val_loss_sup_hse']) 
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--seed', type=int, default=0, help='Base random seed')
parser.add_argument('--feat_str', type=str, default='deg+odeg100') 
parser.add_argument('--height', type=int, default=2)  
parser.add_argument('--aug1', type=str, default='random_mask_permE',
                    choices=['permE', 'maskN', 'maskN_permE', 'none', 'random_mask_permE'])
parser.add_argument('--aug_ratio1', type=float, default=0.3)
parser.add_argument('--aug2', type=str, default='random_mask_permE',
                    choices=['dropN', 'wdropN', 'permE', 'subgraph', 'maskN', 'maskN_permE',
                             'none', 'random4', 'random3', 'random2', 'random_mask_permE'])  
parser.add_argument('--aug_ratio2', type=float, default=0.3)
parser.add_argument('--decay_rate', type=float, default=0.3)  
parser.add_argument('--hypergraph_length_list', type=int, nargs='+', default=[4,6],
                    help='hyperedge length per hierarchy level (e.g. 4 6); default [4, 6]')
parser.add_argument('--use_config_file', dest='use_config_file', action='store_true',
                    help='load hyperparameters from configs/{data_name}.json (default: on); '
                         'explicit CLI flags override JSON values')
parser.add_argument('--no_config_file', dest='use_config_file', action='store_false',
                    help='do not load configs/{data_name}.json; use argparse defaults')
parser.set_defaults(use_config_file=True)
parser.add_argument('--hgsl_constraint', type=str, default='sigmoid',
                    choices=['sigmoid', 'softplus'],
                    help='constraint on learned hypergraph incidence matrix H_T')
parser.add_argument('--drop_zero_incidence', type=int, default=1, choices=[0, 1],
                    help='1: drop exact-zero entries when dense incidence H is converted '
                         'to sparse hyperedge_index; 0 keeps every dense entry, including zeros')
parser.add_argument('--debug', type=int, default=0, choices=[0, 1],
                    help='on AssertionError: print traceback and context, then stop instead of skipping seed')
parser.add_argument('--grad_clip', type=float, default=5.0,
                    help='max gradient norm; 0 disables clipping')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout rate for GNN encoder, classifier MLP, and hypergraph conv (0 disables)')
parser.add_argument('--cluster_assign', type=str, default='em',
                    choices=['em', 'direct_orth'],)
parser.add_argument('--save_checkpoint', action='store_true',
                    help='save per-run best-epoch model checkpoints under '
                         './{checkpoint_dir}/{data_name}/ (default: off)')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                    help='root directory for saved checkpoints (default: checkpoints)')
configs = parser.parse_args()



class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        super().__init__()
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value


def _resolve_num_edges1(configs_dict, num_edges1_cli):
    """Use num_edges2 unless num_edges1 is explicitly set via CLI or a config file."""
    if num_edges1_cli is not None:
        configs_dict['num_edges1'] = num_edges1_cli
    elif configs_dict.get('num_edges1') is None:
        configs_dict['num_edges1'] = configs_dict['num_edges2']


def _normalize_hypergraph_length_list(configs_dict):
    """Coerce hypergraph_length_list to a list of ints."""
    raw = configs_dict.get('hypergraph_length_list')
    if raw is None:
        configs_dict['hypergraph_length_list'] = [4, 6]
    else:
        configs_dict['hypergraph_length_list'] = [int(x) for x in raw]


_CONFIG_FILE_SKIP_KEYS = frozenset({
    'use_config_file', 'save_checkpoint', 'checkpoint_dir',
})


def _cli_explicit_dests(argv=None):
    """Return argparse dest names that were explicitly passed on the command line."""
    if argv is None:
        argv = sys.argv[1:]
    argv_opts = set()
    for token in argv:
        if token.startswith('--'):
            argv_opts.add(token.split('=', 1)[0])
    explicit = set()
    for action in parser._actions:
        if not action.option_strings or action.dest == 'help':
            continue
        if any(opt in argv_opts for opt in action.option_strings):
            explicit.add(action.dest)
    return explicit


def _load_configs(run_args):
    cli_dict = vars(run_args) if not isinstance(run_args, dict) else dict(run_args)
    cli_dict = copy.deepcopy(cli_dict)
    explicit_cli = _cli_explicit_dests()
    configs_dict = copy.deepcopy(cli_dict)
    if (os.path.exists(f'./configs/{configs_dict["data_name"]}.json')
            and cli_dict.get('use_config_file')):
        with open(f'./configs/{configs_dict["data_name"]}.json', 'rt') as f:
            file_cfg = json.load(f)
        for key, value in file_cfg.items():
            if key not in _CONFIG_FILE_SKIP_KEYS:
                configs_dict[key] = value
        for key in explicit_cli:
            if key not in _CONFIG_FILE_SKIP_KEYS:
                configs_dict[key] = cli_dict[key]
    elif not cli_dict.get('use_config_file'):
        print("using default configs")
    num_edges1_cli = cli_dict.get('num_edges1') if 'num_edges1' in explicit_cli else None
    _resolve_num_edges1(configs_dict, num_edges1_cli)
    _normalize_hypergraph_length_list(configs_dict)
    loaded = DotDict(configs_dict)
    loaded.data_root = os.path.normpath(loaded.data_root)
    return loaded


def _release_cuda_memory():
    """Free unreferenced GPU tensors and return cached blocks to the driver."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()


def main():
    """Train one dataset configuration and return the mean test accuracy."""
    loaded = _load_configs(configs)

    os.makedirs(f"./results/{loaded.data_name}", exist_ok=True)
    if loaded.get("save_checkpoint"):
        ckpt_root = loaded.get("checkpoint_dir") or "checkpoints"
        os.makedirs(os.path.join(ckpt_root, loaded.data_name), exist_ok=True)

    acc = None
    exp = None
    try:
        exp = Exp(loaded)
        acc = exp.exp()
    finally:
        if exp is not None:
            exp.cleanup()
            del exp
        _release_cuda_memory()
    if acc is None:
        raise RuntimeError("experiment did not produce an accuracy result")
    return acc


if __name__ == '__main__':
    main()
