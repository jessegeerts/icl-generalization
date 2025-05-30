import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

#Input sequence design

#S is size of training set

#N is number of sample-label pairs in the input sequence

#K is the number of possible labels

#D is the dimensionality of the samples and labels.

#B is burstiness, which controls how many copies of the target are present in the input. N has to be divisible by B. B = 0 chooses randomly.

#p_B is the fraction of bursty sequences.

#P is the probability distribution that a particular label is chosen as the target.
#By default, this is uniform across the K labels.

#The input dimension is 2*N + 1 + D


def get_mus_label_class(K,L,D, seed=None):

    if seed is not None:
        np.random.seed(seed)

    mus_label = np.random.normal(size = (L,D))/np.sqrt(D)
    mus_class = np.random.normal(size = (K,D))/np.sqrt(D)
    if K < L or K%L != 0:
        print("K > L and K%L == 0 is required")
        return 0
    labels_class = np.tile(np.arange(L),int(K/L))

    return mus_label, mus_class, labels_class


def generate_targets_only(mus_label, mus_class, labels_class,S, eps= 0.1, P = None):
    e_fac = 1/np.sqrt(1+eps**2)

    L = mus_label.shape[0]
    K = mus_class.shape[0]
    D = mus_label.shape[1]

    inputs = np.ones((S,D))

    if P is None or len(P) != K:
        P = np.ones(K)/K

    targets = np.random.choice(K, size = S, p = P)

    inputs = e_fac*(mus_class[targets] + eps*np.random.normal(size = (S,D))/np.sqrt(D))

    labels = np.zeros((S,L),dtype= bool)

    for s in range(S):
        labels[s,labels_class[targets[s]]] = True

    return jnp.array(inputs), jnp.array(labels)


def generate_input_seqs(mus_label, mus_class, labels_class,S, N, Nmax, eps= 0.1, B = 0, p_B = 0, P = None, p_C = 0, flip_labels = False, output_target_labels = False, no_repeats = False, shuffle=True):
    e_fac = 1/np.sqrt(1+eps**2)

    L = mus_label.shape[0]
    K = mus_class.shape[0]
    D = mus_label.shape[1]

    K_c = 128  # number of classes to draw from in the fewshot sequences 
    mus_class_new = np.random.normal(size = (K_c,D))/np.sqrt(D)

    if K_c < L or K_c%L != 0:
        print("K > L and K%L == 0 is required")
        return 0
    labels_class_new =  np.tile(np.arange(L),int(K_c/L))

    inputs = np.zeros((S,2*N+1,2*Nmax+1 + D))

    if P is None or len(P) != K:
        P = np.ones(K)/K

    #N has to be divisible by B as we will have N/B copies of each label in the context.
    if (B > 0 and N%B != 0) or B >= N:
        print("N is not divisible by B or N/B is not even or B >= N")
        return 0,0

    if B == 0:
        B = int(N/2)
        p_B = 0

    if L < N and no_repeats:
        raise ValueError('L < N is not possible in no_repeats mode')

    choices = np.zeros((S,int(N/B)), dtype = int)
    if no_repeats:
        for s in range(S):
            label_choices = np.random.choice(np.arange(L), size = (int(N/B)), replace = False)
            pos_choices = np.random.choice(np.arange(int(K/L)), size = (int(N/B)))
            choices[s] = pos_choices*L + label_choices
    else:
        choices = np.random.choice(np.arange(K), size = (S,int(N/B)), p=P)
    choices = np.tile(choices,B)
    if shuffle:  # interleaved condition
        [np.random.shuffle(x) for x in choices]
    else:  # blocked condition
        choices.sort(axis=1)  # make sure that the first N/B samples are the same label, then the next N/B samples are the same label, etc.

    choices_c = np.zeros((S,int(N/B)), dtype = int)
    if no_repeats:
        for s in range(S):
            label_choices = np.random.choice(np.arange(L), size = (int(N/B)), replace = False)
            pos_choices = np.random.choice(np.arange(int(K_c/L)), size = (int(N/B)))
            choices_c[s] = pos_choices*L + label_choices
            #print(choices_c[s], labels_class_new[choices_c[s]],label_choices, pos_choices)
    else:
        choices_c = np.random.choice(np.arange(K_c), size = (S,int(N/B)))
    choices_c = np.tile(choices_c,B)
    [np.random.shuffle(x) for x in choices_c]

    targets_ind = np.random.choice(choices.shape[1], size = (choices.shape[0],))
    targets = choices[np.arange(choices.shape[0]),targets_ind]

    targets_c_ind = np.random.choice(choices_c.shape[1], size = (choices_c.shape[0],))
    targets_c = choices_c[np.arange(choices_c.shape[0]),targets_c_ind]

    filt_B = np.random.uniform(size = S) > p_B

    choices[filt_B] = np.random.choice(K,size  = (np.sum(filt_B),N), p = P)
    targets[filt_B] = np.random.choice(K,size  = (np.sum(filt_B),), p = P)

    filt_C = np.random.uniform(size = S) > p_C

    #print(np.arange(S)[~filt_C])
    #print(np.arange(S)[~filt_B])

    inputs[filt_C,:-1:2,2*Nmax+1:] = (e_fac*(mus_class[choices] + eps*np.random.normal(size = (S,N,D))/np.sqrt(D)))[filt_C]

    if flip_labels:
        wrong_label = (labels_class + 1)%L
        inputs[filt_C,1:-1:2,2*Nmax+1:] = ((mus_label[wrong_label])[choices])[filt_C]
    else:
        inputs[filt_C,1:-1:2,2*Nmax+1:] = ((mus_label[labels_class])[choices])[filt_C]

    inputs[filt_C,-1,2*Nmax+1:] = ((e_fac*(mus_class[targets] + eps*np.random.normal(size = (S,D))/np.sqrt(D))))[filt_C]
    # JPG: for each ~filt_C we fill in new few shot seqs?
    inputs[~filt_C,:-1:2,2*Nmax+1:] = (e_fac*(mus_class_new[choices_c] + eps*np.random.normal(size = (S,N,D))/np.sqrt(D)))[~filt_C]
    inputs[~filt_C,1:-1:2,2*Nmax+1:] = ((mus_label[labels_class_new])[choices_c])[~filt_C]
    inputs[~filt_C,-1,2*Nmax+1:] = (e_fac*(mus_class_new[targets_c] + eps*np.random.normal(size = (S,D))/np.sqrt(D)))[~filt_C]

    shifts = np.random.choice((2*Nmax + 1) - (2*N + 1) + 1, size = (S))

    labels = np.zeros((S,L),dtype= bool)
    target_classes = np.zeros(S, dtype = int)

    for s in range(S):
        if filt_C[s]:
            labels[s,labels_class[targets[s]]] = True
            target_classes[s]= targets[s]
        else:
            labels[s,labels_class_new[targets_c[s]]] = True
            target_classes[s] = -1

        inputs[s,:,shifts[s]:shifts[s] + 2*N+1] = np.identity(2*N+1)

    if output_target_labels:
        return np.array(inputs), jnp.array(labels), target_classes
    else:
        return jnp.array(inputs), jnp.array(labels)


def generate_input_seqs_TI(mus_label, mus_class, labels_class, S, N, Nmax, eps=0.1, B=0, p_B=0, P=None, p_C=0,
                        flip_labels=False, output_target_labels=False, no_repeats=False, shuffle=True, query_pos=None):

    if query_pos is None:
        random_query = True
    else:
        random_query = False

    e_fac = 1 / np.sqrt(1 + eps ** 2)

    L = mus_label.shape[0]  # number of labels. we could assign a different "bigger than" or "smaller than" label for each specific sequence

    K = mus_class.shape[0]
    D = mus_label.shape[1]

    N_items = 7
    N_pairwise = (N_items - 1) * 2
    seq_len = N_pairwise * 2 + 1

    K_c = 128  # number of classes to draw from in the fewshot sequences
    mus_class_new = np.random.normal(size=(K_c, D // 2)) / np.sqrt(D)
    label_mapping_index = np.random.randint(0, L//2, size=(S, 1))

    if K_c < L or K_c % L != 0:
        print("K > L and K%L == 0 is required")
        return 0

    inputs = np.zeros((S, seq_len, 2 * Nmax + 1 + D))

    # item_choices_c = np.random.choice(np.arange(K_c), size=(S, N_items), replace=False)  # here we choose the classes for the few-shot seqs
    item_choices_c = np.array([np.random.choice(np.arange(K_c), size=N_items, replace=False) for _ in range(S)])

    item_1_choices_c = np.concatenate([item_choices_c[:, :-1], item_choices_c[:, 1:]], axis=-1)
    item_2_choices_c = np.concatenate([item_choices_c[:, 1:], item_choices_c[:, :-1]], axis=-1)  # note: these are now always in the same order ,with first all forward pairs and then all reverse pairs
    label_choices_c = np.tile(np.repeat(np.arange(2), N_pairwise//2), (S, 1))

    converted_label_choices_c = np.zeros_like(label_choices_c)
    converted_label_choices_c = np.where((label_choices_c==1), label_mapping_index, converted_label_choices_c)
    converted_label_choices_c = np.where((label_choices_c==0), label_mapping_index + L//2, converted_label_choices_c)

    random_ordering = np.array([np.random.permutation(N_pairwise) for _ in range(S)])
    item_1_choices_c = item_1_choices_c[np.arange(S)[:, None], random_ordering]
    item_2_choices_c = item_2_choices_c[np.arange(S)[:, None], random_ordering]
    label_choices_c = converted_label_choices_c[np.arange(S)[:, None], random_ordering]

    if random_query:
        targets_c_ind = np.random.choice(item_1_choices_c.shape[1], size=(item_1_choices_c.shape[0],))
        targets_c_1 = item_1_choices_c[np.arange(item_1_choices_c.shape[0]), targets_c_ind]
        targets_c_2 = item_2_choices_c[np.arange(item_1_choices_c.shape[0]), targets_c_ind]
    else:
        targets_c_1 =item_choices_c[np.arange(item_choices_c.shape[0]), query_pos[0]]
        targets_c_2 = item_choices_c[np.arange(item_choices_c.shape[0]), query_pos[1]]


    filt_C = np.random.uniform(size=S) > p_C

    # JPG: for each ~filt_C we fill in new few shot seqs?
    inputs[~filt_C, :-1:2, 2 * Nmax + 1:-D//2] = \
    (e_fac * (mus_class_new[item_1_choices_c] + eps * np.random.normal(size=(S, N_pairwise, D//2)) / np.sqrt(D//2)))[~filt_C]
    inputs[~filt_C, :-1:2, 2 * Nmax + 1 + D//2:-1] = \
    (e_fac * (mus_class_new[item_2_choices_c] + eps * np.random.normal(size=(S, N_pairwise, D//2)) / np.sqrt(D//2)))[~filt_C]

    inputs[~filt_C, 1:-1:2, 2 * Nmax + 1:] = ((mus_label[label_choices_c]))[~filt_C]

    inputs[~filt_C, -1, 2 * Nmax + 1:-D//2] = \
    (e_fac * (mus_class_new[targets_c_1] + eps * np.random.normal(size=(S, D//2)) / np.sqrt(D//2)))[~filt_C]
    inputs[~filt_C, -1, 2 * Nmax + 1 + D//2:-1] = \
    (e_fac * (mus_class_new[targets_c_2] + eps * np.random.normal(size=(S, D//2)) / np.sqrt(D//2)))[~filt_C]

    shifts = np.random.choice((2 * Nmax + 1) - seq_len + 1, size=(S))

    labels = np.zeros((S, L), dtype=bool)
    target_classes = np.zeros(S, dtype=int)

    for s in range(S):
        if not filt_C[s]:
            labels[s, label_choices_c[s, targets_c_ind[s]]] = True
            target_classes[s] = -1
        else:
            raise NotImplementedError('This should not happen')

        if shifts[s] + seq_len > 2 * Nmax + 1:
            print('Warning: sequence too long for buffer')
        inputs[s, :, shifts[s]:shifts[s] + seq_len] = np.identity(seq_len)

    # test if sequences is correct (only works for eps=0)
    if eps == 0:
        for s in range(S):
            first_seq = inputs[s][:, 2 * Nmax + 1:]
            if not np.all(first_seq[:-1] == first_seq[-1], axis=1).sum() == 1:
                print('warning: egeg ')
            target_idx = np.all(first_seq[:-1] ==first_seq[-1], axis=1).argmax()
            target_label = np.all(first_seq[target_idx+1] == mus_label, axis=1).argmax()
            if not target_label == labels[s].argmax():
                raise ValueError('Target label not found')

    if output_target_labels:
        return np.array(inputs), jnp.array(labels), target_classes
    else:
        return jnp.array(inputs), jnp.array(labels)