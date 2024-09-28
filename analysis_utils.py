import numpy as np


def extract_pairs_and_outputs(mean_pred_matrix, items):
    pairs = []
    outputs = []

    n_items = mean_pred_matrix.shape[0]

    # Iterate over the upper triangular part of the matrix (excluding the diagonal)
    for i in range(n_items):
        for j in range(n_items):
            if i != j:  # Avoid self-comparisons (diagonal)
                pairs.append((items[i], items[j]))
                outputs.append(mean_pred_matrix[i, j])

    return pairs, outputs


def infer_absolute_ranks(pairs, outputs):
    # Create a list of all unique items and sort them to find the middle
    items = sorted(list(set([item for pair in pairs for item in pair])))
    n_items = len(items)

    # Identify the middle item
    middle_index = n_items // 2
    middle_item = items[middle_index]

    # Mapping of items to indices
    item_to_index = {item: idx for idx, item in enumerate(items)}

    # Set up the linear system: Ax = b
    A = np.zeros((len(outputs), n_items))
    b = np.array(outputs)

    # Populate matrix A based on the pairs and output vector b
    for i, (X, Y) in enumerate(pairs):
        A[i, item_to_index[X]] = 1  # rank(X)
        A[i, item_to_index[Y]] = -1  # rank(Y)

    # Fix the middle item's rank to 0 by removing its column
    A_reduced = np.delete(A, middle_index, axis=1)

    # Solve for the ranks of the other items
    x = np.linalg.lstsq(A_reduced, b, rcond=None)[0]

    # Insert the fixed rank (0) for the middle item back into the ranks array
    ranks = np.insert(x, middle_index, 0)

    # Create a dictionary mapping items to their inferred ranks
    ranks_dict = {item: rank for item, rank in zip(items, ranks)}

    return ranks_dict

# Infer absolute ranks
