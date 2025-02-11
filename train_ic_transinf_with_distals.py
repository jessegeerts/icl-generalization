import wandb
import torch
from train_ti_model_gpu import main
import random
import numpy as np
from configs.config_for_ic_transinf_withdistal import config
import argparse


def update_config_with_args(config, args):
    for arg, value in vars(args).items():
        if value is not None:  # Only override if argument was specified
            # First, check if the key exists at the top level and update if found
            if arg in config.keys():
                config[arg] = value
            else:
                # Otherwise, search recursively in nested dictionaries
                config = find_and_update_nested_key(config, arg, value)
    return config


# Recursive helper to find and update the key in nested config
def find_and_update_nested_key(config, key, value):
    for k, v in config.items():
        if isinstance(v, dict):
            # Recursively search in nested dict
            config[k] = find_and_update_nested_key(v, key, value)
        elif k == key:
            # Update the value if the key matches
            config[k] = value
    return config


# parse arguments to override default config
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--learning_rate', type=float, default=None)
parser.add_argument('--w_decay', type=float, default=None)
parser.add_argument('--save_model', action='store_true', default=None)
parser.add_argument('--no_save_model', action='store_false', dest='save_model')
parser.add_argument('--save_weights', action='store_true', default=None)
parser.add_argument('--no_save_weights', action='store_false', dest='save_weights')
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--n_blocks', type=int, default=None)
parser.add_argument('--n_heads', type=int, default=None)
parser.add_argument('--warmup_steps', type=int, default=None)
parser.add_argument('--ways', type=int, default=None)
parser.add_argument('--shots', type=int, default=None)
parser.add_argument('--use_mlp', action='store_true', default=None)
parser.add_argument('--no_use_mlp', action='store_false', dest='use_mlp')
parser.add_argument('--n_mlp_layers', type=int, default=None)
parser.add_argument('--activation', type=str, default=None)
parser.add_argument('--widening_factor', type=int, default=None)
parser.add_argument('--niters', type=int, default=None)

args = parser.parse_args()

config = update_config_with_args(config, args)

print(config)

torch.random.manual_seed(config['seed'])
random.seed(config['seed'])
np.random.seed(config['seed'])
print(f"Random seed: {config['seed']}")
metrics = main(config, wandb_proj='ic_transinf_with_distals')
wandb.finish()
