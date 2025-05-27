import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


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




def TI_per_pair_plot_with_confidence_intervals(pred_mats_list, ax=None, palette=None, legend_labels=None,
                                               title=None, confidence_level=0.95, **kwargs):
    """
    Plot mean predictions with confidence intervals across multiple runs.

    Args:
        pred_mats_list: List of prediction matrices from different runs
        ax: Matplotlib axis (optional)
        palette: Color palette (optional)
        legend_labels: Labels for legend (optional)
        title: Plot title (optional)
        confidence_level: Confidence level for intervals (default 0.95)
        **kwargs: Additional plotting arguments

    Returns:
        ax: Matplotlib axis object
    """
    # Convert list to array for easier computation
    pred_mats_array = np.array(pred_mats_list)  # Shape: (n_runs, n_models, N, N) or (n_runs, N, N)
    n_runs = len(pred_mats_list)

    # Handle dimensionality
    if pred_mats_array.ndim == 3:  # (n_runs, N, N)
        pred_mats_array = np.expand_dims(pred_mats_array, 1)  # (n_runs, 1, N, N)

    n_runs, n_models, N, _ = pred_mats_array.shape

    # Calculate statistics across runs
    mean_pred_mats = np.mean(pred_mats_array, axis=0)  # (n_models, N, N)
    std_pred_mats = np.std(pred_mats_array, axis=0, ddof=1)  # Sample std

    # Calculate confidence intervals using t-distribution
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha / 2, df=n_runs - 1)
    sem_pred_mats = std_pred_mats / np.sqrt(n_runs)
    ci_half_width = t_critical * sem_pred_mats

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
        mean_distances = {}
        ci_distances = {}

        # Loop through the upper triangular matrix excluding diagonal (symbolic distance 0)
        for i in range(N):
            for j in range(i + 1, N):  # Upper triangle, excluding the diagonal
                dist = j - i  # Symbolic distance
                if dist > 0:  # Ignore symbolic distance 0
                    query_label = get_query_label(i, j)  # Get query label
                    mean_prediction = mean_pred_mats[model, i, j]  # Mean prediction
                    ci_hw = ci_half_width[model, i, j]  # CI half-width

                    # Store prediction in the dictionary, grouped by symbolic distance
                    if dist not in distances:
                        distances[dist] = {'labels': [], 'predictions': []}
                        mean_distances[dist] = {'labels': [], 'predictions': []}
                        ci_distances[dist] = {'labels': [], 'ci_half_widths': []}

                    distances[dist]['labels'].append(query_label)
                    distances[dist]['predictions'].append(mean_prediction)
                    mean_distances[dist]['labels'].append(query_label)
                    mean_distances[dist]['predictions'].append(mean_prediction)
                    ci_distances[dist]['labels'].append(query_label)
                    ci_distances[dist]['ci_half_widths'].append(ci_hw)

        # Initialize a list to hold the x positions for the discontinuous axis
        x_positions = []
        x_ticks = []

        # Variable to manage spacing between groups
        current_x_pos = 0

        # Plot separate lines for each symbolic distance
        for i, dist in enumerate(sorted(distances.keys())):
            labels = distances[dist]['labels']
            predictions = distances[dist]['predictions']
            ci_half_widths = ci_distances[dist]['ci_half_widths']

            # Create a range of x positions for the current symbolic distance
            x_pos = np.arange(current_x_pos, current_x_pos + len(labels))
            x_positions.append(x_pos)
            x_ticks.extend(x_pos)  # Collect the positions for x-ticks

            if legend_labels is not None and i == 0:
                lab = legend_labels[model]
                ci_lab = f'{legend_labels[model]} {int(confidence_level * 100)}% CI'
            else:
                lab = f'Model {model + 1}' if i == 0 else None
                ci_lab = f'Model {model + 1} {int(confidence_level * 100)}% CI' if i == 0 else None

            # Plot the mean line for the current symbolic distance
            line = plt.plot(x_pos, predictions, marker='o', label=lab,
                            color=palette[model], **kwargs)

            # Add confidence intervals as error bars
            plt.errorbar(x_pos, predictions, yerr=ci_half_widths,
                         fmt='none', capsize=3, capthick=1,
                         color=palette[model], alpha=0.7)

            # Alternatively, you can use fill_between for smoother confidence bands
            # (commented out to avoid cluttering, but you can uncomment if preferred)
            # plt.fill_between(x_pos,
            #                 np.array(predictions) - np.array(ci_half_widths),
            #                 np.array(predictions) + np.array(ci_half_widths),
            #                 alpha=0.2, color=palette[model])

            # Update the x position for the next group (to create a gap)
            current_x_pos += len(labels) + 2  # Add a gap of 2 to create space between groups

    # Customize the plot
    if title is None:
        plt.title(f'Mean Model Predictions with {int(confidence_level * 100)}% Confidence Intervals (n={n_runs} runs)')
    else:
        plt.title(title)
    plt.xlabel('Query Pair')
    plt.ylabel('Prediction Value')

    # Set the x-ticks to be the center of each group of query pairs
    plt.xticks(x_ticks, np.concatenate([distances[dist]['labels'] for dist in sorted(distances.keys())]), rotation=90)

    # Add a legend and grid
    # plt.legend(title='Models')
    plt.grid(False)

    # Add statistical information as text
    textstr = f'''Error bars: {int(confidence_level * 100)}% confidence intervals
    Method: t-distribution (df={n_runs - 1})
    Runs: {n_runs} independent experiments
    CI calculation: mean ± t-critical × SEM'''

    # ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
    #         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Tight layout to avoid overlap
    plt.tight_layout()

    # Print statistical summary for reporting
    print(f"Statistical Summary for TI per pair plot:")
    print(f"Number of runs: {n_runs}")
    print(f"Confidence level: {confidence_level * 100}%")
    print(f"Degrees of freedom: {n_runs - 1}")
    print(f"t-critical value: {t_critical:.3f}")
    print(f"Error bars represent: Standard Error of the Mean × t-critical")
    print(f"Assumptions: Normally distributed errors across runs")

    return ax

