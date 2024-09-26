import wandb
from definitions import WANDB_KEY


if __name__ == '__main__':
    from datetime import datetime

    wandb.login(key=WANDB_KEY)

    sweep_configuration = {
        "name": "transinf-icl-sweep-omniglot".format(datetime.now().strftime("%Y%m%d")),
        "method": "random",
        "metric": {"goal": "minimize", "name": "loss"},
        "parameters": {
            "train.learning_rate": {"max": 1e-5, "min": 1e-6, "distribution": "uniform"},
            "train.w_decay": {"max": 1e-4, "min": 1e-7, "distribution": "uniform"},
            "model.n_blocks": {"max": 4, "min": 2, "distribution": "int_uniform"},
            "model.n_heads": {"values": [1, 2, 4, 8], "distribution": "categorical"},
            "train.warmup_steps": {"max": 10000, "min": 3000, "distribution": "int_uniform"},
        },
        "program": "train_ti_model_omniglot.py",
        "project": "ic_transinf_sweep_omniglot",
        "entity": "jesse-geerts-14"
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration)
    print(f"Sweep ID: {sweep_id}")
