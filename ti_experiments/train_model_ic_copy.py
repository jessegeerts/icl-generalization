from ti_experiments.train_model_concat_ti import main
from ti_experiments.configs.cfg_ic_copy import config as default_config

main(default_config, wandb_proj='ic_matchcopy_transinf', seed=42)
