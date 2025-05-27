import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from models import Transformer
from ti_experiments.configs.cfg_iw import config as default_config
from ti_experiments.train_model_concat_ti import eval_at_all_distances
from itertools import product
from sklearn.decomposition import PCA



cfg = default_config
cfg.log.log_to_wandb = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg.model.out_dim = 1
model_dir = '/Users/jessegeerts/Projects/icl-generalization/checkpoints/iw_transinf/uuh8wjk6/'
model_path = os.path.join(model_dir, 'model_5000.pt')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
items = torch.load(os.path.join(model_dir, 'fixed_items.pt'))

model = Transformer(config=default_config.model)
model.load_state_dict(torch.load(model_path))
model.eval()

correct_matrix, holdout_batch, pred_matrix, ranks, model_activations = eval_at_all_distances(cfg,
                                                                                             device,
                                                                                             model,
                                                                                             5000,
                                                                                             get_hiddens=True,
                                                                                             leave_one_out=True,
                                                                                             items=items)

# Example ranks
item_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']  # Replace with your actual ranks

# Generate the labels for off-diagonal elements
labels = []
symb_distance = []
for i, j in product(ranks, ranks):
    if i == j:
        continue  # Skip diagonal elements (AA, BB, etc.)
    label = item_labels[i] + item_labels[j]  # Concatenate to form labels like 'AB', 'AC', etc.
    labels.append(label)
    symb_distance.append(i-j)


len(model_activations)

final_token_activations = [model_activations[i]['hidden_activations'][-1].mean(axis=0)[-1].detach() for i in range(len(model_activations))]
final_token_activations = np.array([np.array(i) for i in final_token_activations])


pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
X_pca = pca.fit_transform(final_token_activations)  # Shape: [P, 2]

plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=symb_distance, cmap='coolwarm', s=100)

# Add labels for each point
for i, inp in enumerate(labels):
    plt.annotate(inp, (X_pca[i, 0], X_pca[i, 1]))

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of final layer Query Token Representations')
plt.grid(True)
plt.show()
