from matplotlib import pyplot as plt
import wandb
from definitions import ATTENTION_CMAP


def log_att_weights(i, out_dict, config):
    for l in range(config.model.n_blocks):
        fig, axes = plt.subplots(1, config.model.n_heads)
        for h in range(config.model.n_heads):
            if config.model.n_heads > 1:
                axes[h].imshow(out_dict[f'block_{l}']['weights'].mean(axis=0)[h].cpu(), cmap=ATTENTION_CMAP)
            else:
                axes.imshow(out_dict[f'block_{l}']['weights'].mean(axis=0)[h].cpu(), cmap=ATTENTION_CMAP)
            if l == 1:
                # plot bar graph of final layer (query) attention weights for one example for each head
                plt.figure()
                plt.bar(range(len(out_dict[f'block_{l}']['weights'][0,-1])), out_dict[f'block_{l}']['weights'][0, -1].cpu())
        if config.log.log_to_wandb:
            wandb.log({f'l{l}_attn_map_icl': fig,
                       'iter': i})  # note: now we're logging the mean of the attention weights across the batch
        plt.close(fig)