import torch
import torch.nn as nn
import numpy as np
from datasets.gauss_datasets import get_mus_label_class


class GaussianEmbedder(nn.Module):
    """A simple Gaussian embedder for the input space of the transformer model.
    Note: this inherits from nn.Module, but it doesn't have any trainable parameters.
    """
    def __init__(self, config):

        super(GaussianEmbedder, self).__init__()
        self.config = config
        self.e_fac = 1 / np.sqrt(1 + config.data.eps ** 2)
        self.Nmax = config.data.Nmax
        self.N = config.seq.N
        self.D = config.data.D
        self.L = config.data.L
        self.S = config.train.batch_size

        self.mus_label, self.mus_class, self.labels_class = self.get_mus_label_class(config.data.K, config.data.L, config.data.D)

    def get_mus_label_class(self, K, L, D):
        n_possible_labels = K
        mus_label = torch.normal(0, 1, size=(n_possible_labels, D)) / np.sqrt(D)
        mus_class = torch.normal(0, 1, size=(K, D)) / np.sqrt(D)
        if K < L or K % L != 0:
            raise ValueError("K >= L and K%L == 0 is required")
        labels_class = torch.tile(torch.arange(L), (1, int(K / L))).squeeze().int()
        return mus_label, mus_class, labels_class

    def forward(self, batch):
        examples = batch['example']
        labels = batch['label']

        inputs = torch.zeros((self.config.train.batch_size, 2 * self.N + 1, 2 * self.Nmax + 1 + self.config.data.D))

        # fill even indices with class examples
        inputs[:, :-1:2, 2 * self.Nmax + 1:] = \
            (self.e_fac * (self.mus_class[examples[:, :-1]] + self.config.data.eps * torch.Tensor(
                np.random.normal(size=(self.S, self.N, self.D))).double() / np.sqrt(self.D)))
        # fill odd indices with label examples
        inputs[:, 1:-1:2, 2 * self.Nmax + 1:] = self.mus_label[labels[:, :-1]]
        # fill last index with target examples
        inputs[:, -1, 2 * self.Nmax + 1:] = \
            (self.e_fac * (self.mus_class[examples[:, -1]] + self.config.data.eps * np.random.normal(size=(self.S, self.D)) / np.sqrt(self.D)))

        # add positional encoding (random shifts)
        shifts = np.random.choice((2 * self.Nmax + 1) - (2 * self.N + 1) + 1, size=(self.S))
        for s in range(self.S):
            inputs[s, :, shifts[s]:shifts[s] + 2 * self.N + 1] = torch.Tensor(np.identity(2 * self.N + 1))

        return inputs

