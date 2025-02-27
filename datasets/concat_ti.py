import torch
import numpy as np
from random import shuffle
import sys
import itertools
import random


def pairwise(iterable):
    # s -> (s0,s1), (s1,s2), (s2, s3), ...
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def generate_iw_sequences_concat_ti(batch_size, num_context_items, item_dim, items=None, distractors=False):

   if items is None:
      raise ValueError("Items must be provided for in-weight sequences")

   if distractors:
      seq_len = (num_context_items - 1) * 2 * 2
   else:
      seq_len = 2

   num_hierarchy_items = items.shape[0]

   batch = torch.zeros(batch_size, seq_len, item_dim * 2)

   for b in range(batch_size):
      if distractors:
         # generate random items with random labels
         distractor_items = torch.randint(0, 2, (num_context_items, item_dim))
         ordering = np.random.permutation(num_context_items)
         pairs = list(pairwise(ordering))

         # Add reversed pairs
         pairs += [(p[1], p[0]) for p in pairs]
         shuffle(pairs)

         idx = 0
         for i, j in pairs:
             pair = torch.cat([distractor_items[i], distractor_items[j]])
             batch[b, idx] = pair

             # Find positions in ordering to determine outcome
             i_pos = np.where(ordering == i)[0][0]
             j_pos = np.where(ordering == j)[0][0]
             outcome = 1 if i_pos < j_pos else -1

             outcome_vec = torch.zeros(item_dim * 2)
             outcome_vec[0] = outcome
             batch[b, idx + 1] = outcome_vec

             idx += 2

      # pick a random adjacent pair of items
      i = random.randint(0, num_hierarchy_items - 2)
      j = i + 1
      # concatenate the items (i then j or j then i 50% of the time)
      if random.random() < 0.5:
         pair = torch.cat([items[i], items[j]])
         outcome = 1
      else:
         pair = torch.cat([items[j], items[i]])
         outcome = -1

      batch[b, -2] = pair
      outcome_vec = torch.zeros(item_dim * 2)
      outcome_vec[0] = outcome
      batch[b, -1] = outcome_vec
   out = {'example': batch[:, :-1, :], 'label': batch[:, -1, 0]}
   return out


def generate_sequences_concat_ti(batch_size, num_items, item_dim, leave_one_out=True):
    if leave_one_out:
        seq_len = (num_items - 1) * 2 * 2
    else:
        seq_len = (num_items - 1) * 2 * 2 + 2
    batch = torch.zeros(batch_size, seq_len, item_dim * 2)

    for b in range(batch_size):
       items = torch.randint(0, 2, (num_items, item_dim))
       ordering = np.random.permutation(num_items)
       pairs = list(pairwise(ordering))

       # Add reversed pairs
       pairs += [(p[1], p[0]) for p in pairs]
       shuffle(pairs)

       context_pairs = []

       idx = 0
       for i, j in pairs:
           pair = torch.cat([items[i], items[j]])
           batch[b, idx] = pair
           context_pairs.append((i, j))

           # Find positions in ordering to determine outcome
           i_pos = np.where(ordering == i)[0][0]
           j_pos = np.where(ordering == j)[0][0]
           outcome = j_pos - i_pos

           outcome_vec = torch.zeros(item_dim * 2)
           outcome_vec[0] = outcome
           batch[b, idx + 1] = outcome_vec

           idx += 2

       if not leave_one_out:
           assert idx == seq_len - 2, "Incorrect sequence length"

       if not leave_one_out:
           # Add the query pair, which during training is one of the context pairs repeated
           query_pair = random.choice(pairs)
           pair = torch.cat([items[query_pair[0]], items[query_pair[1]]])
           batch[b, idx] = pair

           # Find positions in ordering to determine outcome
           i_pos = np.where(ordering == query_pair[0])[0][0]
           j_pos = np.where(ordering == query_pair[1])[0][0]
           outcome = j_pos - i_pos
           outcome_vec = torch.zeros(item_dim * 2)
           outcome_vec[0] = outcome
           batch[b, idx + 1] = outcome_vec

           # assert that the query pair is in the context
           found = False
           for k in range(0, idx, 2):
               if torch.equal(batch[b, k], pair):
                   if torch.equal(batch[b, k+1], outcome_vec):
                       found = True
                   break
           assert found, "Query pair not found in context"

    out = {'example': batch[:, :-1, :], 'label': batch[:, -1, 0]}
    return out


def generate_iw_eval_sequences_concat_ti(batch_size, num_context_items, item_dim, items=None, query_pos=None, distractors=False):
    if items is None:
        raise ValueError("Items must be provided for in-weight sequences")

    if distractors:
        seq_len = (num_context_items - 1) * 2 * 2
    else:
        seq_len = 2

    num_hierarchy_items = items.shape[0]

    if query_pos is None:
        query_pos = np.random.permutation(num_hierarchy_items)[:2]

    if abs(query_pos[1] - query_pos[0]) == 1:
        adjacent_query = True
    else:
        adjacent_query = False

    batch = torch.zeros(batch_size, seq_len, item_dim * 2)

    for b in range(batch_size):
        if distractors:
            # generate random items with random labels
            distractor_items = torch.randint(0, 2, (num_context_items, item_dim))
            ordering = np.random.permutation(num_context_items)
            pairs = list(pairwise(ordering))

            # Add reversed pairs
            pairs += [(p[1], p[0]) for p in pairs]
            shuffle(pairs)

            idx = 0
            for i, j in pairs:
                pair = torch.cat([distractor_items[i], distractor_items[j]])
                batch[b, idx] = pair

                # Find positions in ordering to determine outcome
                i_pos = np.where(ordering == i)[0][0]
                j_pos = np.where(ordering == j)[0][0]
                outcome = 1 if i_pos < j_pos else -1

                outcome_vec = torch.zeros(item_dim * 2)
                outcome_vec[0] = outcome
                batch[b, idx + 1] = outcome_vec

                idx += 2

        i, j = query_pos
        query_pair = torch.cat([items[i], items[j]])
        outcome = j - i
        batch[b, -2] = query_pair
        outcome_vec = torch.zeros(item_dim * 2)
        outcome_vec[0] = outcome
        batch[b, -1] = outcome_vec
    out = {'example': batch[:, :-1, :], 'label': batch[:, -1, 0]}
    return out


def generate_eval_sequences_concat_ti(batch_size, num_items, item_dim, query_pos=None, leave_one_out=True):
   """
   This needs to have the indices of the query item pair as input

   :param batch_size:
   :param num_items:
   :param item_dim:
   :return:
   """

   if query_pos is None:
      query_pos = np.random.permutation(num_items)[:2]

   if abs(query_pos[1] - query_pos[0]) == 1:
      adjacent_query = True
   else:
      adjacent_query = False

   if leave_one_out and adjacent_query:
      seq_len = (num_items - 1) * 2 * 2
   else:
      seq_len = (num_items - 1) * 2 * 2 + 2

   batch = torch.zeros(batch_size, seq_len, item_dim * 2)

   for b in range(batch_size):
       items = torch.randint(0, 2, (num_items, item_dim))
       ordering = np.random.permutation(num_items)

       adjacent_pairs = list(pairwise(ordering))
       # Add reversed pairs
       adjacent_pairs += [(p[1], p[0]) for p in adjacent_pairs]
       # If this is an adjacent query, remove it from context
       query_items = (ordering[query_pos[0]], ordering[query_pos[1]])
       if adjacent_query and leave_one_out:
           adjacent_pairs.remove((query_items[0], query_items[1]))

       shuffle(adjacent_pairs)

       idx = 0
       for i, j in adjacent_pairs:

           pair = torch.cat([items[i], items[j]])
           batch[b, idx] = pair

           # Find positions in ordering to determine outcome
           i_pos = np.where(ordering == i)[0][0]
           j_pos = np.where(ordering == j)[0][0]
           outcome = j_pos - i_pos

           outcome_vec = torch.zeros(item_dim * 2)
           outcome_vec[0] = outcome
           batch[b, idx + 1] = outcome_vec

           idx += 2

       # Add the query pair
       pair = torch.cat([items[ordering[query_pos[0]]], items[ordering[query_pos[1]]]])
       batch[b, idx] = pair
       # Find positions in ordering to determine outcome
       outcome = query_pos[1].item() - query_pos[0].item()
       outcome_vec = torch.zeros(item_dim * 2)
       outcome_vec[0] = outcome
       batch[b, idx + 1] = outcome_vec

   out = {'example': batch[:, :seq_len-1, :], 'label': batch[:, -1, 0]}
   return out


if __name__ == '__main__':
    num_items = 3
    item_dim = 10
    items = torch.randint(0, 2, (num_items, item_dim))
    generate_iw_eval_sequences_concat_ti(1, 3, 10, items=items, distractors=True, query_pos=[0, 2])