import wandb
import torch
from trans_inf_sweep import main
import random
import numpy as np
from configs.working_config_for_ABBA import config

seed = 1

torch.random.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
print(f"Seed: {seed}")
main(config, seq_type='order')
wandb.finish()
