from reddy_replication_torch.experiment import main
from reddy_replication_torch.config import config as default_config
import os

n_runs = 5
for i in range(n_runs):
    print(f"Running iteration {i + 1} with seed {i}")
    cfg = default_config
    cfg.seed = i
    cfg.model_dir = f"models/icl/h{cfg.model.n_heads}_seed_{i}"
    if os.path.exists(cfg.model_dir):
        print(f"Model directory {cfg.model_dir} already exists. Skipping this iteration.")
        continue
    main(cfg)