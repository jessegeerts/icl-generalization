import wandb
import torch
from train_ti_model import main
import random
import numpy as np


for seed in range(100):
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Seed: {seed}")
    main(seq_type='ABBB')
    wandb.finish()