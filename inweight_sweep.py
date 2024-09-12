import wandb
from definitions import WANDB_KEY

if __name__ == '__main__':
    from datetime import datetime

    wandb.login(key=WANDB_KEY)

    sweep_configuration = {
        "name": "transinf-iwl-sweep-{}".format(datetime.now().strftime("%Y%m%d")),
        "method": "random",
        "metric": {"goal": "minimize", "name": "loss"},
        "parameters": {
            "train.learning_rate": {"max": 1e-4, "min": 1e-7, "distribution": "uniform"},
            "train.w_decay": {"max": 1e-4, "min": 1e-7, "distribution": "uniform"},
            "model.n_blocks": {"max": 4, "min": 2, "distribution": "int_uniform"},
            "model.n_heads": {"values": [1, 2, 4, 8], "distribution": "categorical"},
            "train.niters": {"max": 4000, "min": 2000, "distribution": "int_uniform"},
            "seq.N": {"values": [4, 5, 6, 7, 8], "distribution": "categorical"},
        },
        "program": "run_iw_experiment.py",
        "project": "iw_transinf_sweep",
        "entity": "jesse-geerts-14"
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration)
    print(f"Sweep ID: {sweep_id}")
