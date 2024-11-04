# sweep_utils.py
import wandb


def start_run(args):
    sweep_id, main_function, cfg, seq_type = args
    wandb.agent(sweep_id, function=main_function)


