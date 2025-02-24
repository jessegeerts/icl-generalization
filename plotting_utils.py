import wandb
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


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


def TI_per_pair_plot(predict_mat, ax=None, palette=None, legend_labels=None):
    N = predict_mat.shape[-1]

    if predict_mat.ndim == 2:
        predict_mat_plot = np.expand_dims(predict_mat, 0)
    elif predict_mat.ndim == 3:
        predict_mat_plot = predict_mat

    print(predict_mat_plot.ndim)

    n_models = predict_mat_plot.shape[0]

    # Helper function to generate query labels
    def get_query_label(i, j):
        return chr(65 + i) + chr(65 + j)  # Convert indices to characters (A, B, C, ...)

    # Create a figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        plt.sca(ax)

    if palette is None:
        palette = sns.color_palette('rocket', n_models)
    else:
        assert len(palette) >= n_models, f"color palette should contain at least {n_models} colors for {n_models} lines"

    for model in range(n_models):
        # Create a dictionary to store predictions grouped by symbolic distance
        distances = {}

        # Loop through the upper triangular matrix excluding diagonal (symbolic distance 0)
        for i in range(N):
            for j in range(i + 1, N):  # Upper triangle, excluding the diagonal
                dist = j - i  # Symbolic distance
                if dist > 0:  # Ignore symbolic distance 0
                    query_label = get_query_label(i, j)  # Get query label
                    prediction = predict_mat_plot[model, i, j]  # Get prediction value

                    # Store prediction in the dictionary, grouped by symbolic distance
                    if dist not in distances:
                        distances[dist] = {'labels': [], 'predictions': []}
                    distances[dist]['labels'].append(query_label)
                    distances[dist]['predictions'].append(prediction)

        # Initialize a list to hold the x positions for the discontinuous axis
        x_positions = []
        x_ticks = []

        # Variable to manage spacing between groups
        current_x_pos = 0

        # Plot separate lines for each symbolic distance
        for i, dist in enumerate(sorted(distances.keys())):
            labels = distances[dist]['labels']
            predictions = distances[dist]['predictions']

            # Create a range of x positions for the current symbolic distance
            x_pos = np.arange(current_x_pos, current_x_pos + len(labels))
            x_positions.append(x_pos)
            x_ticks.extend(x_pos)  # Collect the positions for x-ticks

            if legend_labels is not None:
                lab = legend_labels[i]
            else:
                lab = None
            # Plot the line for the current symbolic distance
            plt.plot(x_pos, predictions, marker='o', label=lab, color=palette[model])

            # Update the x position for the next group (to create a gap)
            current_x_pos += len(labels) + 2  # Add a gap of 2 to create space between groups

    # Customize the plot
    plt.title('Model Predictions by Symbolic Distance')
    plt.xlabel('Query Pair')
    plt.ylabel('Prediction Value')

    # Set the x-ticks to be the center of each group of query pairs
    plt.xticks(x_ticks, np.concatenate([distances[dist]['labels'] for dist in sorted(distances.keys())]), rotation=90)

    # Add a legend and grid
    if legend_labels is not None:
        plt.legend(title='Symbolic Distance')
    plt.grid(False)

    # Tight layout to avoid overlap
    plt.tight_layout()

    return ax