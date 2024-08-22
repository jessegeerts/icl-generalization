from configs.working_config_for_ABBA import config
from input_embedders import GaussianEmbedderForOrdering
import torch

config.model.pos_emb_randomization = 'per_sequence'
config.seq.N = 3

emb = GaussianEmbedderForOrdering(config)

example = torch.zeros((config.train.batch_size, 2 * config.seq.N + 2), dtype=torch.int)
label = torch.zeros((config.train.batch_size, 1 * config.seq.N + 1), dtype=torch.int)
emb.forward({'example': example, 'label': label})