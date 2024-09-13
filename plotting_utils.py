import wandb
from matplotlib import pyplot as plt


def plot_and_log_matrix(cfg, matrix, iter, xticks, yticks, cmap, vmin, vmax, title=None):
    fig = plt.figure()
    ax = plt.imshow(matrix, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.xticks(xticks)
    plt.yticks(yticks)
    if title is not None:
        plt.title(title)
    plt.colorbar()
    plt.close(fig)
    # Log the figures as images in wandb
    if cfg.log.log_to_wandb:
        wandb.log({title: wandb.Image(fig), 'iter': iter})
    return fig, ax