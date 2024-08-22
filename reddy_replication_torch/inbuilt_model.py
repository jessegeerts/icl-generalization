"""Pytorch's inbuilt transformer model, for benchmarking my own.
"""
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer


class TorchTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model=config.h_dim,
                                    nhead=config.n_heads,
                                    dim_feedforward=config.widening_factor * config.h_dim,
                                    batch_first=True,
                                    dropout=config.drop_p), num_layers=config.n_blocks)
        self.proj_head = nn.Linear(config.h_dim, config.out_dim)

    def forward(self, x):
        # h = embed_stimuli_and_labels(stimuli, labels)
        h = x
        mask = Transformer.generate_square_subsequent_mask(h.shape[1])
        # convert mask to boolean (-inf should go to false and 0 to True)
        mask = mask == 0
        h = self.transformer(h, is_causal=True, mask=mask)
        pred = self.proj_head(h)
        return pred[:, -1, :]  # Select the output corresponding to the last stimulus (query stimulus)
