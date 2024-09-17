from experiment_transitive_inference import run_experiment
import torch

torch.set_num_threads(4)

if __name__ == '__main__':
    run_experiment()
