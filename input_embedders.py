import torch
import torch.nn as nn
import numpy as np
from datasets.gauss_datasets import get_mus_label_class
from models import get_sinusoidal_positional_embeddings_2
import h5py as h5


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


class GaussianEmbedderForOrdering(nn.Module):
    """A simple Gaussian embedder for the input space of the transformer model.
    Note: this inherits from nn.Module, but it doesn't have any trainable parameters.
    """
    def __init__(self, config, device):

        super(GaussianEmbedderForOrdering, self).__init__()
        self.config = config
        self.e_fac = 1 / np.sqrt(1 + config.data.eps ** 2)
        self.Nmax = config.data.Nmax
        self.N = config.seq.N
        self.D = config.data.D
        self.L = config.data.L
        self.S = config.train.batch_size
        self.pos_embedding_type = config.model.pos_emb_type
        self.device = device
        if config.model.pos_emb_type == 'sinusoidal':
            self.positional_embedding = get_sinusoidal_positional_embeddings_2(config.model.max_T * 3, self.D).to(device)

        self.mus_label, self.mus_class, self.labels_class = self.get_mus_label_class(config.data.K, config.data.L, config.data.D)
        self.mus_label = self.mus_label.to(device)
        self.mus_class = self.mus_class.to(device)
        self.labels_class = self.labels_class.to(device)

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
        if self.config.seq.include_flipped:
            n_flips = 2
        else:
            n_flips = 1
        seq_len = (self.N-1) * 3 * n_flips + 2  # N-1 combos, 3 items per combo, 2 orderings, 2 targets
        n_example_pairs = (self.N-1) * n_flips + 1  # this includes the target
        inputs = torch.zeros((self.config.train.batch_size, seq_len, 2 * self.Nmax + self.config.data.D))

        # fill every first 2 indices with class examples
        inputs[:, ::3, 2 * self.Nmax:] = \
            (self.e_fac * (self.mus_class[examples[:, ::2]] + self.config.data.eps * torch.Tensor(
                np.random.normal(size=(self.S, n_example_pairs, self.D))).double().to(self.device) / np.sqrt(self.D)))
        inputs[:, 1::3, 2 * self.Nmax:] = \
            (self.e_fac * (self.mus_class[examples[:, 1::2]] + self.config.data.eps * torch.Tensor(
                np.random.normal(size=(self.S, n_example_pairs, self.D))).double().to(self.device) / np.sqrt(self.D)))
        # fill every 3rd index with label examples
        inputs[:, 2:-2:3, 2 * self.Nmax:] = self.mus_label[labels[:, :-1]]

        # add positional encoding (random shifts)
        if self.config.model.add_pos_encodings:
            shifts = np.random.choice((2 * self.Nmax + 2) - (2 * self.N + 2) + 1, size=(self.S))
            for s in range(self.S):
                if self.config.model.pos_emb_randomization == 'per_batch':
                    shift_choice = 0
                    write_to_example = s
                elif self.config.model.pos_emb_randomization == 'per_sequence':
                    shift_choice = s
                    write_to_example = s
                elif self.config.model.pos_emb_randomization == 'only_first':
                    shift_choice = 0
                    write_to_example = 0
                elif self.config.model.pos_emb_randomization == 'only_last':
                    shift_choice = 0
                    write_to_example = self.S - 1
                elif self.config.model.pos_emb_randomization == 'no_shift':
                    shifts = np.zeros(self.S, dtype=int)
                    shift_choice = 0
                    write_to_example = s
                else:
                    raise ValueError('Invalid positional embedding randomization: {}'.format(
                        self.config.model.pos_emb_randomization))

                if self.pos_embedding_type == 'onehot':
                    inputs[write_to_example, :,
                    shifts[shift_choice]:shifts[shift_choice] + seq_len] = torch.Tensor(
                        np.identity(seq_len)).to(self.device)
                else:
                    inputs[write_to_example, :, :2 * self.Nmax] = self.positional_embedding[0,
                                                                  shifts[shift_choice]: shifts[
                                                                                            shift_choice] + seq_len]
        return inputs.to(self.device)


class OmniglotEmbedder(nn.Module):
    """A simple Omniglot embedder for the input space of the transformer model.
    Note: this inherits from nn.Module, but it doesn't have any trainable parameters.
    """
    def __init__(self, config, device):
        super(OmniglotEmbedder, self).__init__()
        self.config = config
        # embedidngs are stored in a file. this class simply reads them
        embeddings_file = 'datasets/omniglot_resnet18_randomized_order_s0.h5'
        with h5.File(embeddings_file, 'r') as f:
            embeddings = torch.Tensor(np.array(f['resnet18/224/feat']))
        # we only take the first exemplar from each class, for now
        embeddings = embeddings[:, 0, :]
        self.pos_embedding_type = config.model.pos_emb_type
        num_classes, emb_dim = embeddings.shape
        assert emb_dim == config.data.D
        self.embeddings = embeddings.to(device)
        self.D = config.data.D  # dimension of the embeddings (512 in the omniglot case)
        self.Nmax = config.data.Nmax
        self.N = config.seq.N
        self.L = config.data.L
        self.label_embeddings = (torch.normal(0, 1, size=(self.L, self.D)) / np.sqrt(self.D)).to(device)
        self.S = config.train.batch_size
        self.device = device

    def forward(self, batch):
        examples = batch['example']
        labels = batch['label']

        seq_len = examples.shape[1] + labels.shape[1] - 1
        inputs = torch.zeros((self.config.train.batch_size, seq_len, self.D + 2 * self.Nmax)).to(self.device)

        # fill every first 2 indices with class examples
        inputs[:, ::3, 2 * self.Nmax:] = self.embeddings[examples[:, ::2]]
        inputs[:, 1::3, 2 * self.Nmax:] = self.embeddings[examples[:, 1::2]]
        # fill every 3rd index with label examples (except the last one)
        inputs[:, 2::3, 2 * self.Nmax:] = self.label_embeddings[labels[:, :-1]]

        # add positional encoding (random shifts)
        if self.config.model.add_pos_encodings:
            shifts = np.random.choice((2 * self.Nmax + 2) - (2 * self.N + 2) + 1, size=(self.S))
            for s in range(self.S):
                if self.config.model.pos_emb_randomization == 'per_batch':
                    shift_choice = 0
                    write_to_example = s
                elif self.config.model.pos_emb_randomization == 'per_sequence':
                    shift_choice = s
                    write_to_example = s
                elif self.config.model.pos_emb_randomization == 'only_first':
                    shift_choice = 0
                    write_to_example = 0
                elif self.config.model.pos_emb_randomization == 'only_last':
                    shift_choice = 0
                    write_to_example = self.S - 1
                elif self.config.model.pos_emb_randomization == 'no_shift':
                    shifts = np.zeros(self.S, dtype=int)
                    shift_choice = 0
                    write_to_example = s
                else:
                    raise ValueError('Invalid positional embedding randomization: {}'.format(
                        self.config.model.pos_emb_randomization))

                if self.pos_embedding_type == 'onehot':
                    inputs[write_to_example, :,
                    shifts[shift_choice]:shifts[shift_choice] + seq_len] = torch.Tensor(
                        np.identity(seq_len)).to(self.device)
                else:
                    inputs[write_to_example, :, :2 * self.Nmax] = self.positional_embedding[0,
                                                                  shifts[shift_choice]: shifts[
                                                                                            shift_choice] + seq_len]

        return inputs.to(self.device)


