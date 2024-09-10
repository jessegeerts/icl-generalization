import wandb
import torch
from train_ti_model_gpu import main
import random
import numpy as np
from configs.working_config_for_ABBA import config

seed = 1

torch.random.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
print(f"Seed: {seed}")
main(config, seq_type=config.seq.train_seq_type)
wandb.finish()
