"""
Generate partial exposure sequences as described in Chan et al (2022).

TODO: these are now embedded as gaussians but it would be better to pass
the raw data to the input embedder and let it do the embedding.
"""

import torch
import numpy as np
import random


def get_partial_exposure_sequence(config, mus_label):
    """
    Generate a sequence of stimuli and labels for the partial exposure paradigm:

    context                                             query
    AW → 0 AX → 0 BW → 1 BW → 1 CY → 2 CZ → 2           BX → ?

    Each subvector has some subvector class mean. stimuli are concatenated subvectors. The query is an unseen
    combination of subvectors. These are generated in a fewshot way the subvector class means are generated randomly
    each time.
    """
    Nmax = config.data.Nmax
    S = 1
    N = config.seq.N

    e_fac = 1 / np.sqrt(1 + config.data.eps ** 2)

    first_subvector_names = ['A', 'B', 'C']
    second_subvector_names = ['W', 'X', 'Y', 'Z']
    mus_first_subvectors = np.random.normal(size=(len(first_subvector_names), config.data.subD)) / np.sqrt(config.data.subD)
    mus_second_subvectors = np.random.normal(size=(len(second_subvector_names), config.data.subD)) / np.sqrt(config.data.subD)
    sub1_dict = {k: v for k, v in zip(first_subvector_names, mus_first_subvectors)}
    sub2_dict = {k: v for k, v in zip(second_subvector_names, mus_second_subvectors)}
    class_names = ['AW', 'AX', 'BW', 'BW', 'CY', 'CZ', 'BX']
    mus_class = np.array([np.concatenate([sub1_dict[c[0]], sub2_dict[c[1]]]) for c in class_names])

    # mus_label = np.random.normal(size=(config.data.L, config.data.D)) / np.sqrt(config.data.D)
    labels_class = [0, 0, 1, 1, 2, 2, 0]
    # now get the choices for the context. basically each class except BX chosen twice
    choices_c = list(range(len(class_names) - 1)) * 2
    random.shuffle(choices_c)

    inputs = np.zeros((1, 2 * config.seq.N + 1, 2 * config.data.Nmax + 1 + config.data.D))

    inputs[:, :-1:2, 2 * config.data.Nmax + 1:] = \
    (e_fac * (mus_class[choices_c] + config.data.eps * np.random.normal(size=(1, config.seq.N, config.data.D)) / np.sqrt(config.data.D)))
    inputs[:, 1:-1:2, 2 * config.data.Nmax + 1:] = ((mus_label[labels_class])[choices_c])
    inputs[:, -1, 2 * config.data.Nmax + 1:] = \
    (e_fac * (mus_class[-1] + config.data.eps * np.random.normal(size=(1, config.data.D)) / np.sqrt(config.data.D)))

    # add positional encoding

    shifts = np.random.choice((2 * Nmax + 1) - (2 * N + 1) + 1, size=(S))
    for s in range(S):
        inputs[s, :, shifts[s]:shifts[s] + 2 * N + 1] = torch.Tensor(np.identity(2 * N + 1))

    return inputs, np.array(class_names)[choices_c], np.array(labels_class)[choices_c]


def exemplar_strategy(stim, labels, query):
    """Exemplar strategy for classification.

    This is a simple strategy that classifies the query stimulus as the label of the most similar stimulus in the
    sequence.
    """
    similarity = query @ stim.T
    max_similar = torch.argmax(similarity)
    max_label = labels[max_similar]
    return max_label

