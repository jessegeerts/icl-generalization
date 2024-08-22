import wandb
import torch
from trans_inf_sweep import main
import random
import numpy as np
from configs.oneshot_config import config


seed = 1

torch.random.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
print(f"Seed: {seed}")
main(seq_type='ABBA')
wandb.finish()
