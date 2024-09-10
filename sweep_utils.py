# sweep_utils.py
import wandb


def start_run(args):
    sweep_id, main_function, cfg, seq_type = args
    wandb.agent(sweep_id, function=main_function)


def update_nested_config(config, update):
    for key, value in update.items():
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return config
