from ti_experiments.train_model_concat_ti import main
from ti_experiments.configs.cfg_iw import config as default_config
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
import torch
import os
from models import Transformer
from ti_experiments.train_model_concat_ti import eval_at_all_distances

sns.set_context("paper", font_scale=2)


n_runs = 10

model_paths = []
all_metrics = []
for i in range(n_runs):
    print(f"Running iteration {i + 1} with seed {i}")
    metrics = main(default_config, wandb_proj='iw_transinf', seed=i)
    all_metrics.append(metrics)
    model_paths.append(metrics['model_path'])

print(model_paths)

iters = np.arange(0, default_config.train.niters, default_config.log.logging_interval)

fig, ax = plt.subplots()
for i in range(n_runs):
    plt.plot(iters, all_metrics[i]['loss'], label=f'Run {i + 1}')
plt.ylabel('Loss')
plt.xlabel('Training steps')
plt.tight_layout()
# plt.savefig('loss.png')
plt.show()

fig, ax = plt.subplots()
for i in range(n_runs):
    plt.plot(iters, all_metrics[i]['holdout_accuracy'], label=f'Run {i + 1}')

plt.ylabel('Accuracy (adjacent queries)')
plt.xlabel('Training steps')
plt.tight_layout()
# plt.savefig('accuracy.png')
plt.show()
