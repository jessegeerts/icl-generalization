import wandb
import torch
from train_ti_model import main
import random
import numpy as np
from configs.config_for_ic_transinf import config

seed = 1

torch.random.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
print(f"Seed: {seed}")
metrics = main(config, seq_type=config.seq.train_seq_type)
wandb.finish()
