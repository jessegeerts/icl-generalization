"""train in-context TI model, but instead of outputting 1 or -1 and training
with MSE, we ouput "higher" or "lower" and train with cross-entropy loss
"""
import torch
from train_ti_model_gpu import main
from configs.config_for_ti_ic_class import config

torch.set_num_threads(4)

output = main(config=config, wandb_proj="in-context-TI-omniglot-ic-class")