import torch
from train_ti_model_gpu import main
from configs.config_for_ic_transinf_omniglot import config

torch.set_num_threads(4)


if __name__ == '__main__':
    import os
    import pandas as pd

    save_dir = f'results/{config.seq.train_type}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    metrics = main(config=config, wandb_proj="in-context-TI-omniglot")

    acc_df = pd.DataFrame(metrics['accuracies']).rename(
        columns={d: f'mean_accuracy_at_abs(distance)_{d}' for d in metrics['accuracies'][0].keys()})
    pred_df = pd.DataFrame(metrics['predictions']).rename(
        columns={d: f'mean_prediction_at_distance_{d}' for d in metrics['predictions'][0].keys()})

    # get run number by checking how many files that start with 'metrics_run_' are in the directory
    i = len([f for f in os.listdir(save_dir) if f.startswith('metrics_run_')])

    metrics_df = pd.concat([acc_df, pred_df], axis=1).assign(run=i)
    metrics_df.to_csv(os.path.join(save_dir, f'metrics_run_{i}.csv'), index=False)

