import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import matplotlib.pyplot as plt
import os

from reddy_replication_torch.model import Transformer
from reddy_replication_torch.inbuilt_model import TorchTransformer
from datasets.reddy.datasets_v2 import *
from definitions import WANDB_KEY, ATTENTION_CMAP
from utils import dotdict as dd


def generate_input_seqs_TI_fixed(mus_label, mus_class, labels_class, S, N, Nmax, eps=0.1, B=0, p_B=0, P=None, p_C=0,
                                 flip_labels=False, output_target_labels=False, no_repeats=False, shuffle=True,
                                 query_pos=None):
    """Fixed version of generate_input_seqs_TI that handles query_pos correctly."""

    if query_pos is None:
        random_query = True
    else:
        random_query = False

    e_fac = 1 / np.sqrt(1 + eps ** 2)

    L = mus_label.shape[0]  # number of labels
    K = mus_class.shape[0]
    D = mus_label.shape[1]

    N_items = 7
    N_pairwise = (N_items - 1) * 2
    seq_len = N_pairwise * 2 + 1

    K_c = 128  # number of classes to draw from in the fewshot sequences
    mus_class_new = np.random.normal(size=(K_c, D // 2)) / np.sqrt(D)
    label_mapping_index = np.random.randint(0, L // 2, size=(S, 1))

    if K_c < L or K_c % L != 0:
        print("K > L and K%L == 0 is required")
        return 0

    inputs = np.zeros((S, seq_len, 2 * Nmax + 1 + D))

    item_choices_c = np.array([np.random.choice(np.arange(K_c), size=N_items, replace=False) for _ in range(S)])

    item_1_choices_c = np.concatenate([item_choices_c[:, :-1], item_choices_c[:, 1:]], axis=-1)
    item_2_choices_c = np.concatenate([item_choices_c[:, 1:], item_choices_c[:, :-1]], axis=-1)
    label_choices_c = np.tile(np.repeat(np.arange(2), N_pairwise // 2), (S, 1))

    converted_label_choices_c = np.zeros_like(label_choices_c)
    converted_label_choices_c = np.where((label_choices_c == 1), label_mapping_index, converted_label_choices_c)
    converted_label_choices_c = np.where((label_choices_c == 0), label_mapping_index + L // 2,
                                         converted_label_choices_c)

    random_ordering = np.array([np.random.permutation(N_pairwise) for _ in range(S)])
    item_1_choices_c = item_1_choices_c[np.arange(S)[:, None], random_ordering]
    item_2_choices_c = item_2_choices_c[np.arange(S)[:, None], random_ordering]
    label_choices_c = converted_label_choices_c[np.arange(S)[:, None], random_ordering]

    if random_query:
        targets_c_ind = np.random.choice(item_1_choices_c.shape[1], size=(item_1_choices_c.shape[0],))
        targets_c_1 = item_1_choices_c[np.arange(item_1_choices_c.shape[0]), targets_c_ind]
        targets_c_2 = item_2_choices_c[np.arange(item_1_choices_c.shape[0]), targets_c_ind]
    else:
        # Get the actual items at the specified positions
        targets_c_1 = item_choices_c[np.arange(item_choices_c.shape[0]), query_pos[0]]
        targets_c_2 = item_choices_c[np.arange(item_choices_c.shape[0]), query_pos[1]]
        targets_c_ind = None

    filt_C = np.random.uniform(size=S) > p_C

    # Fill in context pairs
    inputs[~filt_C, :-1:2, 2 * Nmax + 1:-D // 2] = \
        (e_fac * (mus_class_new[item_1_choices_c] + eps * np.random.normal(size=(S, N_pairwise, D // 2)) / np.sqrt(
            D // 2)))[~filt_C]
    inputs[~filt_C, :-1:2, 2 * Nmax + 1 + D // 2:-1] = \
        (e_fac * (mus_class_new[item_2_choices_c] + eps * np.random.normal(size=(S, N_pairwise, D // 2)) / np.sqrt(
            D // 2)))[~filt_C]

    inputs[~filt_C, 1:-1:2, 2 * Nmax + 1:] = ((mus_label[label_choices_c]))[~filt_C]

    # Fill in query pair
    inputs[~filt_C, -1, 2 * Nmax + 1:-D // 2] = \
        (e_fac * (mus_class_new[targets_c_1] + eps * np.random.normal(size=(S, D // 2)) / np.sqrt(D // 2)))[~filt_C]
    inputs[~filt_C, -1, 2 * Nmax + 1 + D // 2:-1] = \
        (e_fac * (mus_class_new[targets_c_2] + eps * np.random.normal(size=(S, D // 2)) / np.sqrt(D // 2)))[~filt_C]

    shifts = np.random.choice((2 * Nmax + 1) - seq_len + 1, size=(S))

    labels = np.zeros((S, L), dtype=bool)
    target_classes = np.zeros(S, dtype=int)

    for s in range(S):
        if not filt_C[s]:
            if random_query:
                labels[s, label_choices_c[s, targets_c_ind[s]]] = True
            else:
                pos1, pos2 = query_pos[0][s], query_pos[1][s]
                if pos1 < pos2:
                    labels[s, label_mapping_index[s, 0] + L // 2] = True
                else:
                    labels[s, label_mapping_index[s, 0]] = True
            target_classes[s] = -1
        else:
            raise NotImplementedError('This should not happen')

        if shifts[s] + seq_len > 2 * Nmax + 1:
            print('Warning: sequence too long for buffer')
        inputs[s, :, shifts[s]:shifts[s] + seq_len] = np.identity(seq_len)

    if output_target_labels:
        return np.array(inputs), np.array(labels), target_classes
    else:
        return np.array(inputs), np.array(labels)


def sample_query_positions_by_distance(batch_size, n_items=7):
    """Sample query pairs with controlled distance distribution."""
    pos1 = np.zeros(batch_size, dtype=int)
    pos2 = np.zeros(batch_size, dtype=int)

    for i in range(batch_size):
        pos1[i] = np.random.randint(0, n_items)
        pos2[i] = np.random.choice([j for j in range(n_items) if j != pos1[i]])

    return (pos1, pos2)


def eval_loss_and_accuracy(mod, inputs, labels, criterion, save_weights=False):
    y_hat, out_dict = mod(inputs, save_weights=save_weights)
    loss = criterion(y_hat, torch.argmax(labels.float(), dim=-1))
    predicted_labels = torch.argmax(y_hat, dim=1)
    accuracy = (predicted_labels == torch.argmax(labels.float(), dim=-1)).float().mean()
    return loss, accuracy, out_dict


def calculate_induction_strength(cfg, holdout_batch, n, out_dict_eval):
    ind_strngth = []
    batch_wo_pos_code = holdout_batch[:, :, -cfg.data.D:]
    contexts = batch_wo_pos_code[:, :-1, :]
    queries = batch_wo_pos_code[:, -1, :]
    dotprod = torch.einsum('btd,bd->bt', contexts, queries)
    most_similar = dotprod.argmax(dim=-1)
    correct_ids = most_similar + 1

    for h in range(cfg.model.n_heads):
        attn_weights = out_dict_eval[f'block_1']['weights'][:, h, :, :]
        query_attn = attn_weights[:, -1]
        batch_indices = torch.arange(holdout_batch.shape[0], device=query_attn.device)
        correct_attention = query_attn[batch_indices, correct_ids]
        mask = torch.ones_like(query_attn, dtype=torch.bool)
        mask[batch_indices, correct_ids] = False
        non_correct_values = query_attn[mask].reshape(holdout_batch.shape[0], -1)
        non_correct_means = non_correct_values.mean(dim=1)
        induction_strength = correct_attention.mean() - non_correct_means.mean()
        ind_strngth.append(induction_strength.item())
    return ind_strngth


def main_curriculum():
    """Main training function with curriculum learning: start with ICL, gradually introduce TI."""

    cfg = dd(dict(
        model=dd(dict(
            h_dim=128,
            n_heads=8,
            n_blocks=2,
            include_mlp=[False] * 2,
            softmax_attn=[True] * 2,
            activation='relu',
            n_mlp_layers=None,
            apply_ln=False,
            widening_factor=1,
            max_T=64,
            out_dim=None,
            drop_p=0.0,
            dampening=1.
        )),
        data=dd(dict(
            S=10000,
            K=2 ** 10,
            L=32,
            D=63,
            alpha=0.,
            eps=0,
        )),
        seq=dd(dict(
            N=12,
            Nmax=32,
            B=1,
            pB=1.,
            pC=1.,
            shuf=True,
            train_type='curriculum',
            no_repeats=False
        )),
        train=dd(dict(
            batch_size=128,
            learning_rate=.01,
            learning_rate_adam=1e-3,
            w_decay=1e-7,
            niters=10000 * 10,
            optim='adam',
            curriculum_warmup=20000,  # Pure ICL for first 20k iterations
            curriculum_transition=40000,  # Linear transition from 20k-40k iterations
        )),
        log_to_wandb=True,
        logging_interval=500,
        save_weights=True,
        save_model=False,
        saving_interval=4000,
        model_dir='models/icl',
        seed=1,
    ))

    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    S = cfg.data.S
    K = cfg.data.K
    L = cfg.data.L
    D = cfg.data.D
    alpha = cfg.data.alpha
    eps = cfg.data.eps
    N = cfg.seq.N
    B = cfg.seq.B
    pB = cfg.seq.pB
    pC = cfg.seq.pC
    Nmax = cfg.seq.Nmax

    p_class = 1.0 / (np.arange(1, K + 1) ** alpha)
    p_class /= np.sum(p_class)
    no_repeats = cfg.seq.no_repeats

    experiment_name = 'TI_curriculum_I{}_warmup{}_trans{}_K{}_N{}_L{}_D{}_lr{}_adam'.format(
        cfg.train.niters,
        cfg.train.curriculum_warmup,
        cfg.train.curriculum_transition,
        cfg.data.K,
        cfg.seq.N,
        cfg.data.L,
        cfg.data.D,
        cfg.train.learning_rate_adam,
    )
    cfg.model.out_dim = cfg.data.L
    print(experiment_name)

    if cfg.log_to_wandb:
        wandb.login(key=WANDB_KEY)
        wandb.init(project="reddy-replication", name=experiment_name, config=cfg)

    mus_label, mus_class, labels_class = get_mus_label_class(K, L, D, seed=0)

    # Test sets
    test_inputs_TI, test_labels_TI = generate_input_seqs_TI(
        mus_label, mus_class, labels_class, S, N, Nmax,
        eps=eps, B=B, p_B=pB, p_C=pC, no_repeats=no_repeats
    )
    test_inputs_ic, test_labels_ic = generate_input_seqs(
        mus_label, mus_class, labels_class, S, N, Nmax,
        eps=eps, P=p_class, B=B, p_B=1, p_C=1, no_repeats=no_repeats
    )
    test_inputs_iw, test_labels_iw = generate_input_seqs(
        mus_label, mus_class, labels_class, S, N, Nmax,
        eps=eps, P=p_class, B=0, p_B=0, p_C=0, no_repeats=no_repeats
    )

    test_inputs_ic = torch.from_numpy(np.array(test_inputs_ic)).float().to(device)
    test_inputs_iw = torch.from_numpy(np.array(test_inputs_iw)).float().to(device)
    test_labels_ic = torch.from_numpy(np.array(test_labels_ic)).to(device)
    test_labels_iw = torch.from_numpy(np.array(test_labels_iw)).to(device)
    test_inputs_TI = torch.from_numpy(np.array(test_inputs_TI)).float().to(device)
    test_labels_TI = torch.from_numpy(np.array(test_labels_TI)).to(device)

    model = Transformer(config=cfg.model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate_adam, weight_decay=cfg.train.w_decay)
    criterion = nn.CrossEntropyLoss()

    # Training loop with curriculum
    for n in range(cfg.train.niters):
        model.train()

        # Curriculum schedule: compute probability of TI vs ICL
        if n < cfg.train.curriculum_warmup:
            # Phase 1: Pure ICL
            p_ti = 0.0
        elif n < cfg.train.curriculum_transition:
            # Phase 2: Linear transition from 0% to 100% TI
            progress = (n - cfg.train.curriculum_warmup) / (
                        cfg.train.curriculum_transition - cfg.train.curriculum_warmup)
            p_ti = progress
        else:
            # Phase 3: Pure TI
            p_ti = 1.0

        if np.random.rand() < p_ti:
            # TI training with varied distances
            query_positions = sample_query_positions_by_distance(cfg.train.batch_size, n_items=7)
            inputs_batch, labels_batch, target_classes = generate_input_seqs_TI_fixed(
                mus_label, mus_class, labels_class,
                cfg.train.batch_size, N, Nmax,
                eps=eps, P=p_class, B=B, p_B=pB, p_C=pC,
                output_target_labels=True, no_repeats=no_repeats,
                query_pos=query_positions
            )
        else:
            # ICL training
            inputs_batch, labels_batch, target_classes = generate_input_seqs(
                mus_label, mus_class, labels_class,
                cfg.train.batch_size, N, Nmax,
                eps=eps, P=p_class, B=B, p_B=pB, p_C=pC,
                output_target_labels=True, no_repeats=no_repeats
            )

        inputs_batch = torch.from_numpy(inputs_batch).float().to(device)
        labels_batch = torch.from_numpy(np.array(labels_batch)).to(device)

        optimizer.zero_grad()
        y_hat, out_dict = model(inputs_batch)
        loss = criterion(y_hat, torch.argmax(labels_batch.float(), dim=-1))
        loss.backward()
        optimizer.step()

        if n % cfg.logging_interval == 0:
            model.eval()
            with torch.no_grad():
                if cfg.log_to_wandb:
                    wandb.log({'train_loss': loss.item(), 'iter': n, 'p_ti': p_ti})

                ti_loss, ti_accuracy, out_dict_TI = eval_loss_and_accuracy(
                    model, test_inputs_TI, test_labels_TI, criterion, save_weights=cfg.save_weights
                )
                icl_loss, icl_accuracy, out_dict = eval_loss_and_accuracy(
                    model, test_inputs_ic, test_labels_ic, criterion, save_weights=False
                )
                iwl_loss, iwl_accuracy, out_dict = eval_loss_and_accuracy(
                    model, test_inputs_iw, test_labels_iw, criterion, save_weights=False
                )

                if cfg.log_to_wandb:
                    wandb.log({
                        'ti_loss': ti_loss.item(), 'ti_accuracy': ti_accuracy.item(),
                        'icl_loss': icl_loss.item(), 'icl_accuracy': icl_accuracy.item(),
                        'iwl_loss': iwl_loss.item(), 'iwl_accuracy': iwl_accuracy.item(),
                        'iter': n
                    })

                print(f'iter {n}, p_ti: {p_ti:.2f}, loss: {loss.item():.4f}, '
                      f'ti: {ti_accuracy.item():.4f}, ic: {icl_accuracy.item():.4f}, iw: {iwl_accuracy.item():.4f}')

    if cfg.log_to_wandb:
        wandb.finish()


if __name__ == '__main__':
    main_curriculum()
