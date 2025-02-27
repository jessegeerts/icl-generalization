from ti_experiments.train_model_concat_ti import main
from ti_experiments.configs.cfg_iw import config as default_config

main(default_config, wandb_proj='iw_transinf', seed=42)
