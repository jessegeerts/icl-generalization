import wandb
import torch
from train_ti_model_gpu import main
import random
import numpy as np
from configs.config_for_ic_transinf_withdistal import config

seed = 1

torch.random.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
print(f"Seed: {seed}")
metrics = main(config, wandb_proj='ic_transinf_with_distals')
wandb.finish()
