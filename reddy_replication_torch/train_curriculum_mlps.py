import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import matplotlib.pyplot as plt
import os
import argparse

from reddy_replication_torch.model import Transformer
from reddy_replication_torch.inbuilt_model import TorchTransformer
from datasets.reddy.datasets_v2 import *
from definitions import WANDB_KEY, ATTENTION_CMAP
from utils import dotdict as dd
from plotting_utils import TI_per_pair_plot


def generate_input_seqs_TI_fixed(mus_label, mus_class, labels_class, S, N, Nmax, eps=0.1, B=0, p_B=0, P=None, p_C=0,
                                 flip_labels=False, output_target_labels=False, no_repeats=False, shuffle=True,
                                 query_pos=None):
    """Fixed version of generate_input_seqs_TI that handles query_pos correctly."""

    if query_pos is None:
        random_query_from_context = True
    else:
        random_query_from_context = False

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

    if random_query_from_context:
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
            if random_query_from_context:
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

    return pos1, pos2


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


def parse_args():
    """Parse command line arguments to override default config."""
    parser = argparse.ArgumentParser(description="Train transformer with curriculum learning on TI task")

    # Model architecture
    parser.add_argument("--n_blocks", type=int, default=2, help="Number of transformer blocks")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--h_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--add_mlp", action="store_true", help="Add MLP layers to all blocks")
    parser.add_argument("--mlp_blocks", type=str, default=None,
                        help="Comma-separated list of which blocks get MLPs (e.g., '0,1' or '1')")
    parser.add_argument("--widening_factor", type=int, default=4, help="MLP widening factor")
    parser.add_argument("--apply_ln", action="store_true", help="Apply layer normalization")
    parser.add_argument("--drop_p", type=float, default=0.0, help="Dropout probability")

    # Training
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for Adam")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--niters", type=int, default=100000, help="Number of training iterations")
    parser.add_argument("--curriculum_warmup", type=int, default=20000, help="Pure ICL iterations")
    parser.add_argument("--curriculum_transition", type=int, default=40000, help="End of transition period")
    parser.add_argument("--w_decay", type=float, default=1e-7, help="Weight decay")
    parser.add_argument("--include_distal_queries", action="store_true", help="Include distal queries during TI training")

    # Data
    parser.add_argument("--eps", type=float, default=0.0, help="Within-class variance")

    # Logging
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--logging_interval", type=int, default=500, help="Logging frequency")
    parser.add_argument("--save_weights", action="store_true", help="Save attention weights")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name override")

    # Misc
    parser.add_argument("--seed", type=int, default=1, help="Random seed")

    return parser.parse_args()


def main_curriculum():
    """Main training function with curriculum learning: start with ICL, gradually introduce TI."""

    args = parse_args()

    # Determine MLP configuration
    if args.mlp_blocks is not None:
        # Parse comma-separated list
        mlp_block_indices = [int(x) for x in args.mlp_blocks.split(',')]
        include_mlp = [i in mlp_block_indices for i in range(args.n_blocks)]
    elif args.add_mlp:
        # All blocks get MLP
        include_mlp = [True] * args.n_blocks
    else:
        # No MLPs
        include_mlp = [False] * args.n_blocks

    cfg = dd(dict(
        model=dd(dict(
            h_dim=args.h_dim,
            n_heads=args.n_heads,
            n_blocks=args.n_blocks,
            include_mlp=include_mlp,
            softmax_attn=[True] * args.n_blocks,
            activation='relu',
            n_mlp_layers=None,
            apply_ln=args.apply_ln,
            widening_factor=args.widening_factor,
            max_T=64,
            out_dim=None,
            drop_p=args.drop_p,
            dampening=1.
        )),
        data=dd(dict(
            S=10000,
            K=2 ** 10,
            L=32,
            D=63,
            alpha=0.,
            eps=args.eps,
        )),
        seq=dd(dict(
            N=12,
            Nmax=32,
            B=1,
            pB=1.,
            pC=1.,
            shuf=True,
            train_type='curriculum',
            no_repeats=False,
        )),
        train=dd(dict(
            batch_size=args.batch_size,
            learning_rate=.01,
            learning_rate_adam=args.learning_rate,
            w_decay=args.w_decay,
            niters=args.niters,
            optim='adam',
            curriculum_warmup=args.curriculum_warmup,
            curriculum_transition=args.curriculum_transition,
            include_distal_queries=args.include_distal_queries
        )),
        log_to_wandb=not args.no_wandb,
        logging_interval=args.logging_interval,
        save_weights=args.save_weights,
        save_model=False,
        save_figs=True,
        saving_interval=4000,
        model_dir='models/icl',
        seed=args.seed,
    ))

    # Create experiment name
    mlp_str = 'mlp_' + ''.join([str(i) for i, x in enumerate(include_mlp) if x]) if any(include_mlp) else 'nomlp'
    include_distal_str = 'distal' if args.include_distal_queries else 'nodistal'
    if args.exp_name is not None:
        experiment_name = args.exp_name
    else:
        experiment_name = 'TI_curriculum_{}_bl{}_{}_{}_warmup{}_trans{}_lr{}_eps{}'.format(
            include_distal_str,
            args.n_blocks,
            mlp_str,
            'ln' if args.apply_ln else 'noln',
            args.curriculum_warmup,
            args.curriculum_transition,
            args.learning_rate,
            args.eps,
        )
    cfg.model.out_dim = cfg.data.L
    print(experiment_name)
    print(f"Model config: {args.n_blocks} blocks, include_mlp={include_mlp}, "
          f"widening_factor={args.widening_factor}, ln={args.apply_ln}")

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

    if cfg.log_to_wandb:
        wandb.login(key=WANDB_KEY)
        wandb.init(project="reddy-replication", name=experiment_name, config=cfg)

    mus_label, mus_class, labels_class = get_mus_label_class(K, L, D, seed=0)

    # Test sets for all possible query pairs
    test_inputs_by_pair = {}
    test_labels_by_pair = {}

    n_items = 7
    for pos1 in range(n_items):
        for pos2 in range(n_items):
            if pos1 == pos2:
                continue  # Skip same-item pairs

            pair_name = f"{pos1}{pos2}"

            # Create query positions with fixed pair
            query_pos_fixed = (
                np.full(S, pos1, dtype=int),
                np.full(S, pos2, dtype=int)
            )

            test_inputs_pair, test_labels_pair = generate_input_seqs_TI_fixed(
                mus_label, mus_class, labels_class, S, N, Nmax,
                eps=eps, B=B, p_B=pB, p_C=pC, no_repeats=no_repeats,
                query_pos=query_pos_fixed
            )
            test_inputs_by_pair[pair_name] = torch.from_numpy(np.array(test_inputs_pair)).float().to(device)
            test_labels_by_pair[pair_name] = torch.from_numpy(np.array(test_labels_pair)).to(device)

    # Original test sets
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
            if cfg.train.include_distal_queries:
                # for now, equal sampling of query distances
                query_positions = sample_query_positions_by_distance(cfg.train.batch_size, n_items=7)
            else:
                query_positions = None
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

                # Evaluate on each query pair
                pair_accuracies = {}
                for pos1 in range(7):
                    for pos2 in range(7):
                        if pos1 == pos2:
                            continue
                        pair_name = f"{pos1}{pos2}"
                        pair_loss, pair_accuracy, _ = eval_loss_and_accuracy(
                            model, test_inputs_by_pair[pair_name],
                            test_labels_by_pair[pair_name], criterion, save_weights=False
                        )
                        pair_accuracies[pair_name] = pair_accuracy.item()
                        if cfg.log_to_wandb:
                            wandb.log({f'ti_accuracy_pair_{pair_name}': pair_accuracy.item(), 'iter': n})

                # put these in a matrix pred_matrix for visualization
                pred_matrix = np.zeros((7, 7))
                for pos1 in range(7):
                    for pos2 in range(7):
                        if pos1 == pos2:
                            pred_matrix[pos1, pos2] = np.nan
                        else:
                            pair_name = f"{pos1}{pos2}"
                            pred_matrix[pos1, pos2] = pair_accuracies[pair_name]

                # Visualize prediction accuracy matrix with TI_per_pair_plot
                fig, ax = plt.subplots(figsize=(6, 5))
                TI_per_pair_plot(pred_matrix, ax)
                ax.set_ylabel('Accuracy')
                ax.set_title(f'Test Accuracy by Query Pair at Iter {n}')
                if cfg.log_to_wandb:
                    wandb.log({"ti_accuracy_pair_plot": wandb.Image(fig), 'iter': n})
                # save to disk
                if cfg.save_figs:
                    fig_dir = os.path.join('figures', 'reddy_replication', experiment_name)
                    os.makedirs(fig_dir, exist_ok=True)
                    fig_path = os.path.join(fig_dir, f'ti_accuracy_pair_plot_iter_{n}.pdf')
                    fig.savefig(fig_path)
                plt.close(fig)

                # Compute distance-based aggregates for summary
                distance_accuracies = {d: [] for d in range(1, 7)}
                for pos1 in range(7):
                    for pos2 in range(7):
                        if pos1 == pos2:
                            continue
                        distance = abs(pos2 - pos1)
                        pair_name = f"{pos1}{pos2}"
                        distance_accuracies[distance].append(pair_accuracies[pair_name])

                # Average per distance
                avg_distance_accuracies = {d: np.mean(accs) for d, accs in distance_accuracies.items()}
                if cfg.log_to_wandb:
                    for distance, avg_acc in avg_distance_accuracies.items():
                        wandb.log({f'ti_accuracy_dist_{distance}_avg': avg_acc, 'iter': n})

                # Original evaluations
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

                # Print with distance breakdown
                dist_str = ', '.join([f'd{d}:{avg_distance_accuracies[d]:.3f}' for d in range(1, 7)])
                print(f'iter {n}, p_ti: {p_ti:.2f}, loss: {loss.item():.4f}, '
                      f'ti: {ti_accuracy.item():.4f}, ic: {icl_accuracy.item():.4f}, iw: {iwl_accuracy.item():.4f}')
                print(f'  avg by distance: {dist_str}')

                # Optionally print a few specific pairs
                if n % (cfg.logging_interval * 10) == 0:  # Less frequent detailed output
                    print(f'  sample pairs: 01:{pair_accuracies["01"]:.3f}, 06:{pair_accuracies["06"]:.3f}, '
                          f'12:{pair_accuracies["12"]:.3f}, 36:{pair_accuracies["36"]:.3f}')

    if cfg.log_to_wandb:
        wandb.finish()


if __name__ == '__main__':
    main_curriculum()
