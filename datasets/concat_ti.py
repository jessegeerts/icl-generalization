import torch
import numpy as np
from random import shuffle
import sys
import itertools


def pairwise(iterable):
    # s -> (s0,s1), (s1,s2), (s2, s3), ...
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def generate_sequences_concat_ti(batch_size, num_items, item_dim):
    seq_len = (num_items - 1) * 2 * 2
    batch = torch.zeros(batch_size, seq_len, item_dim * 2)

    for b in range(batch_size):
       items = torch.randint(0, 2, (num_items, item_dim))
       ordering = np.random.permutation(num_items)
       pairs = list(pairwise(ordering))

       # Add reversed pairs
       pairs += [(p[1], p[0]) for p in pairs]
       shuffle(pairs)

       idx = 0
       for i, j in pairs:
           pair = torch.cat([items[i], items[j]])
           batch[b, idx] = pair

           # Find positions in ordering to determine outcome
           i_pos = np.where(ordering == i)[0][0]
           j_pos = np.where(ordering == j)[0][0]
           outcome = 1 if i_pos < j_pos else -1

           outcome_vec = torch.zeros(item_dim * 2)
           outcome_vec[0] = outcome
           batch[b, idx + 1] = outcome_vec

           idx += 2
    out = {'example': batch[:, :seq_len-1, :], 'label': batch[:, -1, 0]}
    return out


def generate_eval_sequences_concat_ti(batch_size, num_items, item_dim, query=None):
   """
   This needs to have the indices of the query item pair as input

   :param batch_size:
   :param num_items:
   :param item_dim:
   :return:
   """
   seq_len = (num_items - 1) * 2 * 2 + 2  # these are all adjacent pairs + the query pair
   batch = torch.zeros(batch_size, seq_len, item_dim * 2)

   if query is None:
      query = np.random.permutation(num_items)[:2]

   for b in range(batch_size):
       items = torch.randint(0, 2, (num_items, item_dim))
       ordering = np.random.permutation(num_items)
       adjacent_pairs = list(pairwise(ordering))

       # Add reversed pairs
       adjacent_pairs += [(p[1], p[0]) for p in adjacent_pairs]
       shuffle(adjacent_pairs)

       idx = 0
       for i, j in adjacent_pairs:
           pair = torch.cat([items[i], items[j]])
           batch[b, idx] = pair

           # Find positions in ordering to determine outcome
           i_pos = np.where(ordering == i)[0][0]
           j_pos = np.where(ordering == j)[0][0]
           outcome = 1 if i_pos < j_pos else -1

           outcome_vec = torch.zeros(item_dim * 2)
           outcome_vec[0] = outcome
           batch[b, idx + 1] = outcome_vec

           idx += 2

       # Add the query pair
       pair = torch.cat([items[query[0]], items[query[1]]])
       batch[b, idx] = pair
       # Find positions in ordering to determine outcome
       i_pos = np.where(ordering == query[0].item())[0][0]
       j_pos = np.where(ordering == query[1].item())[0][0]
       outcome = 1 if i_pos < j_pos else -1

       outcome_vec = torch.zeros(item_dim * 2)
       outcome_vec[0] = outcome
       batch[b, idx + 1] = outcome_vec

   out = {'example': batch[:, :seq_len-1, :], 'label': batch[:, -1, 0]}
   return out
