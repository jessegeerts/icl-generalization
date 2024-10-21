"""This module contains helper methods for analyzing the transformer's
embeddings using dimensionality reduction.
"""

import numpy as np
import torch
from models import Transformer

from experiment_transitive_inference import TransInfSeqGen, get_transitive_inference_sequence_embeddings
import itertools


def load_from_checkpoint(model_path, config):
    model = Transformer(config=config.model)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def get_output_and_activations_for_TI(model, config, fixed_classes,
                                      layer=-1, token_id=-1, n_evals=50):
    pred_matrix = np.zeros((len(fixed_classes), len(fixed_classes)))

    activation = []

    seqgen = TransInfSeqGen(config)
    ranks = np.arange(len(fixed_classes))
    for i, j in itertools.product(ranks, ranks):
        if i == j:
            continue

        query = (fixed_classes[i], fixed_classes[j])
        yhats = []
        last_activations = []
        for n in range(n_evals):
            context = seqgen.get_random_context()
            inputs = get_transitive_inference_sequence_embeddings(context, query)
            target = 0 if i == j else 1 if i < j else -1
            inputs = inputs.unsqueeze(0)
            y_hat, out_dict = model(inputs, save_weights=False, save_hidden_activations=True)

            activation_final = out_dict['hidden_activations'][layer].squeeze()[token_id]
            last_activations.append(activation_final)

            pred = torch.sign(y_hat).item()
            correct = int(pred == target)
            yhats.append(y_hat.item())
        pred_matrix[i, j] = np.mean(yhats)

        # mean of last activations
        activation.append(torch.stack(last_activations).mean(dim=0))
    return pred_matrix, activation
