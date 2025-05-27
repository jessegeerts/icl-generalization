from ti_experiments.train_model_concat_ti import main
from ti_experiments.configs.cfg_iw import config as default_config
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_context("paper", font_scale=2)

metrics = main(default_config, wandb_proj='iw_transinf', seed=42)

iters = np.arange(0, default_config.train.niters, default_config.log.logging_interval)

fig, ax = plt.subplots()
plt.plot(iters, metrics['loss'])
plt.ylabel('Loss')
plt.xlabel('Training steps')
plt.tight_layout()
plt.savefig('loss.png')
plt.show()

fig, ax = plt.subplots()
plt.plot(iters, metrics['holdout_accuracy'])
plt.ylabel('Accuracy (adjacent queries)')
plt.xlabel('Training steps')
plt.tight_layout()
plt.savefig('accuracy.png')
plt.show()
