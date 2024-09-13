import torch
from train_ti_model_gpu import main
from configs.config_for_ic_transinf_omniglot import config

torch.set_num_threads(4)


if __name__ == '__main__':
    main(config=config, wandb_proj="in-context-TI-omniglot")
