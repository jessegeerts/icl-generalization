import wandb
from definitions import WANDB_KEY


if __name__ == '__main__':
    from datetime import datetime

    wandb.login(key=WANDB_KEY)

    sweep_configuration = {
        "name": "transinf-icl-sweep-macbook".format(datetime.now().strftime("%Y%m%d")),
        "method": "random",
        "metric": {"goal": "minimize", "name": "loss"},
        "parameters": {
            "train.learning_rate": {"max": 1e-5, "min": 1e-7, "distribution": "uniform"},
            "train.w_decay": {"max": 1e-4, "min": 1e-7, "distribution": "uniform"},
            "model.n_blocks": {"max": 16, "min": 2, "distribution": "int_uniform"},
            "model.n_heads": {"values": [1, 2, 4, 8], "distribution": "categorical"},
            "train.warmup_steps": {"max": 10000, "min": 3000, "distribution": "int_uniform"},
            "seq.ways": {"values": [2, 3, 4, 5, 6, 7, 8], "distribution": "categorical"},
        },
        "program": "train_ti_model_gpu.py",
        "project": "ic_transinf_sweep",
        "entity": "jesse-geerts-14"
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration)
    print(f"Sweep ID: {sweep_id}")
