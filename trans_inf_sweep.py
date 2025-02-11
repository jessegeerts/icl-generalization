import wandb
from definitions import WANDB_KEY


if __name__ == '__main__':
    from datetime import datetime

    wandb.login(key=WANDB_KEY)

    sweep_configuration = {
        "name": "transinf-icl-sweep-distals".format(datetime.now().strftime("%Y%m%d")),
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "loss"},
        "parameters": {
            "train.learning_rate": {"max": 7.e-04, "min": 1e-7, "distribution": "uniform"},
            "train.w_decay": {"max": 1e-4, "min": 1e-7, "distribution": "uniform"},
            "model.use_mlp": {"values": [True, False], "distribution": "categorical"},
            "model.n_heads": {"values": [1, 2, 4, 8], "distribution": "categorical"},
            "train.warmup_steps": {"max": 10000, "min": 3000, "distribution": "int_uniform"},
            "train.niters": {"max": 500000, "min": 100000, "distribution": "int_uniform"},
        },
        "program": "train_ic_transinf_with_distals.py",
        "project": "ic_transinf_with_distals",
        "entity": "jesse-geerts-14"
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration)
    print(f"Sweep ID: {sweep_id}")
