
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import wandb
import math
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from itertools import product

# from configs.oneshot_config import config as default_config
from configs.config_for_ic_transinf import config as default_config
from datasets.data_generators import SymbolicDatasetForSampling, TransInfSeqGenerator, GaussianDataset
from input_embedders import GaussianEmbedderForOrdering
from main_utils import log_att_weights
from models import Transformer
from definitions import WANDB_KEY, COLOR_PALETTE
from utils import MyIterableDataset
from utils import dotdict as dd

cp = sns.color_palette(COLOR_PALETTE)


def eval_loss_and_accuracy(mod, inputs, labels, criterion, config):
    y_hat, out_dict = mod(inputs, save_weights=config.save_weights)

    if config.model.prediction_mode == 'regress':
        labels = labels.float()
        labels[labels == 0] = -1
        y_hat = y_hat.squeeze()

    loss = criterion(y_hat, labels)
    if config.model.prediction_mode == 'classify':
        predicted_labels = torch.argmax(y_hat, dim=1)
    else:
        predicted_labels = torch.sign(y_hat)
    accuracy = (predicted_labels == labels).float().mean()

    if config.log.log_to_wandb:

        A = out_dict['block_1']['weights'][:, :, -1]
        # todo: this is outdated. we can instead split on whether the query is the same item order
        label_matches_query = inputs['label'][:, :-1]==inputs['label'][:,-1].unsqueeze(1)
        n_labels_in_context = label_matches_query.shape[1]
        for i in range(n_labels_in_context):
            accuracy_i = (predicted_labels == labels)[label_matches_query[:,i]].float().mean()
            wandb.log({'accuracy_when_label_{}'.format(i): accuracy_i.item()})

            # log attention to each label
            attention_to_label = A[torch.arange(128), :, i*3+2]
            wandb.log({'attention_to_label_{}'.format(i): attention_to_label.mean().item()})


        # log attention distribution as histogram for when "true" label is the first or second label
        att_dist = [A[label_matches_query[:,i], :, :].mean(axis=0).mean(axis=0) for i in range(n_labels_in_context)]

        if config.seq.include_flipped:
            n_flips = 2
        else:
            n_flips = 1
        fig, ax = plt.subplots()
        x = torch.arange(len(att_dist[0]))
        for i, att_dist_i in enumerate(att_dist):
            ax.bar(x + i * 0.2, att_dist_i, width=0.2, color=cp[i], align='center', label='Attention distribution when {}th label is correct'.format(i+1))
        xticks =  (['img'] * 2 + ['lab']) * (config.seq.N-1) * n_flips + ['img'] * 2
        ax.set_xticks(x+.2, xticks)
        # Adding labels and title
        plt.title('Attention distribution when first or second label is correct')
        plt.legend()

        wandb.log({"attention_distribution": wandb.Image(fig)})
        plt.close(fig)
    return loss, accuracy, out_dict


def update_nested_config(config, update):
    for key, value in update.items():
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return config


def main():

    run = wandb.init()
    cfg = default_config.copy()

    sweep_params = dict(run.config)  # Get sweep parameters from wandb
    cfg = update_nested_config(cfg, sweep_params)  # Merge sweep params into the default config
    cfg = dd(cfg)

    device = 'cpu'  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    metrics = {
        'holdout_accuracy': [],
        'predictions': [],
        'loss': [],
        'accuracies': []
    }

    cfg.seq.N = cfg.seq.ways * cfg.seq.shots

    if cfg.model.prediction_mode == 'classify':
        cfg.model.out_dim = cfg.data.L
    else:
        cfg.model.out_dim = 1  # for regression

    ### load or construct the dataset
    if cfg.data.type == 'gaussian':
        dataset = GaussianDataset(cfg.data.K, cfg.data.L, cfg.data.D)
    else:
        dataset = SymbolicDatasetForSampling(cfg.data.K)

    seqgen = TransInfSeqGenerator(dataset)

    if cfg.seq.seq_type == 'order':
        # todo: we can swap this for "in-weight" sequences with constant mapping
        train_generator = partial(seqgen.get_fewshot_order_seq, cfg.seq.ways, cfg.seq.shots, mode='train', train_distal=cfg.seq.include_distal_in_training)
        holdout_generator = partial(seqgen.get_fewshot_order_seq, cfg.seq.ways, cfg.seq.shots, mode='test')
        # fixme: this is just a placeholder for now
        iwl_generator = seqgen.get_fewshot_order_seq(cfg.seq.ways, cfg.seq.shots)
    elif cfg.seq.seq_type == 'ABBB':
        train_generator = seqgen.get_AB_BB_seqs(cfg.seq.shots)
        holdout_generator = seqgen.get_AB_BB_seqs(cfg.seq.shots)
        iwl_generator = seqgen.get_AB_BB_seqs(cfg.seq.shots)
    elif cfg.seq.seq_type == 'ABBA':
        train_generator = seqgen.get_AB_BA_seqs(cfg.seq.shots, set='train')
        holdout_generator = seqgen.get_AB_BA_seqs(cfg.seq.shots, set='test')
        iwl_generator = seqgen.get_AB_BA_seqs(cfg.seq.shots, set='all')
    else:
        raise ValueError('Invalid sequence type: {}'.format(cfg.seq.seq_type))

    iterdataset = MyIterableDataset(train_generator, holdout_generator, iwl_generator)
    dataloader = DataLoader(iterdataset, batch_size=cfg.train.batch_size)

    # prepare model
    input_embedder = GaussianEmbedderForOrdering(cfg)
    model = Transformer(config=cfg.model, input_embedder=input_embedder).to(device)  # my custom transformer encoder

    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.w_decay)
    if cfg.train.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.niters, eta_min=.00001)
    elif cfg.train.lr_scheduler == 'none':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.)
    elif cfg.train.lr_scheduler == 'warmup_cosine':
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min((step + 1) / cfg.train.warmup_steps, 1.0) * 0.5 * (
                        1 + math.cos(step / cfg.train.niters * math.pi))
        )
    elif cfg.train.lr_scheduler == 'warmup_linear':
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min((step + 1) / cfg.train.warmup_steps, 1.0) * (1 - step / cfg.train.niters)
        )
    elif cfg.train.lr_scheduler == 'warmup_constant':
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min((step + 1) / cfg.train.warmup_steps, 1.0)
        )
    else:
        raise ValueError('Invalid learning rate scheduler: {}'.format(cfg.train.lr_scheduler))

    if cfg.model.prediction_mode == 'classify':
        criterion = nn.CrossEntropyLoss()
    elif cfg.model.prediction_mode == 'regress':
        criterion = nn.MSELoss()
    else:
        raise ValueError('Invalid prediction mode: {}'.format(cfg.model.prediction_mode)
                         + 'Valid options are: classify, regress')

    steps_above_criterion = 0
    iterdataset.set_mode('train')
    iterator = iter(dataloader)
    for n in range(cfg.train.niters):
        model.train()
        batch = next(iterator)

        optimizer.zero_grad()

        # for the transformer encoder, we need to reshape the input to (seq_len, batch_size, emb_dim)
        y_hat, _ = model(batch)

        loss = criterion(y_hat, batch['label'][:, -1].float().view(-1, 1))
        loss.backward()
        optimizer.step()
        scheduler.step()

        if n % cfg.log.logging_interval == 0:
            print(f'iteration {n}, loss {loss}')
            if cfg.log.log_to_wandb:
                # log current loss
                wandb.log({'loss': loss.item(), 'iter': n})
                # log current learning rate
                for param_group in optimizer.param_groups:
                    wandb.log({'lr': param_group['lr'], 'iter': n})

            # evaluate on holdout set (still with adjacent pairs)
            iterdataset.set_mode('holdout')
            iterator = iter(dataloader)
            model.eval()
            holdout_batch = next(iterator)
            holdout_loss, holdout_accuracy, out_dict_eval = eval_loss_and_accuracy(model, holdout_batch, holdout_batch['label'][:, -1].long(), criterion, cfg)
            print(f'holdout loss: {holdout_loss}, holdout accuracy: {holdout_accuracy}')
            if cfg.log.log_to_wandb:
                wandb.log({'holdout_loss': holdout_loss.item(), 'holdout_accuracy': holdout_accuracy.item(), 'iter': n})
            metrics['holdout_accuracy'].append(holdout_accuracy.item())
            metrics['loss'].append(loss.item())
            if cfg.save_weights:
                log_att_weights(n, out_dict_eval, cfg)

            # evaluate on non-adjacent pairs (dist == 2 etc )
            for dist in range(-cfg.seq.N+1, cfg.seq.N):
                iterdataset.set_mode('holdout', eval_distance=dist)
                iterator = iter(dataloader)
                model.eval()
                holdout_batch = next(iterator)
                y_hat, _ = model(holdout_batch)
                predicted_labels = torch.sign(y_hat.squeeze())
                true_label_sign = torch.sign(holdout_batch['label'][:, -1].float())
                accuracy = (predicted_labels == true_label_sign).float().mean()
                print(f'holdout accuracy for distance {dist}: {accuracy}')
                output_mean = y_hat.mean()

                if cfg.log.log_to_wandb:
                    wandb.log({'holdout_accuracy_dist_{}'.format(dist): accuracy.item(), 'iter': n})
                    wandb.log({'output_mean_dist_{}'.format(dist): output_mean.item(), 'iter': n})

            # todo: this should be per rank because it might be different for the edge cases, for example
            correct_matrix = torch.zeros((cfg.seq.N, cfg.seq.N))
            pred_matrix = torch.zeros((cfg.seq.N, cfg.seq.N))

            ranks = torch.arange(cfg.seq.N)
            for i, j in product(ranks, ranks):
                iterdataset.set_mode('holdout', set_query_ranks=(i, j))
                iterator = iter(dataloader)
                model.eval()
                holdout_batch = next(iterator)
                y_hat, _ = model(holdout_batch)
                predicted_labels = torch.sign(y_hat.squeeze())
                true_label_sign = torch.sign(holdout_batch['label'][:, -1].float())
                accuracy = (predicted_labels == true_label_sign).float().mean()
                output_mean = y_hat.detach().mean()
                correct_matrix[i, j] = accuracy
                pred_matrix[i, j] = output_mean

            # Create a figure for the correct matrix
            fig_correct = plt.figure()
            plt.imshow(correct_matrix, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
            plt.xticks(ranks)
            plt.yticks(ranks)
            plt.title('Correct Matrix')
            plt.colorbar()
            plt.close(fig_correct)  # Close the figure to prevent it from displaying in your Python environment

            # Create a figure for the pred matrix
            fig_pred = plt.figure()
            plt.imshow(pred_matrix, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)
            plt.xticks(ranks)
            plt.yticks(ranks)
            plt.title('Pred Matrix')
            plt.colorbar()
            plt.close(fig_pred)  # Close the figure to prevent it from displaying in your Python environment

            # Log the figures as images in wandb
            if cfg.log.log_to_wandb:
                wandb.log({"correct_matrix": wandb.Image(fig_correct), "pred_matrix": wandb.Image(fig_pred), 'iter': n})

            # Initialize a dictionary to store the mean accuracies for each absolute distance
            mean_accuracies = {}
            mean_preds = {}
            # Calculate the mean accuracy and output for each distance
            for distance in range(-cfg.seq.N+1, cfg.seq.N):
                # Get the elements in the diagonal at the current absolute distance
                diagonal_elements = torch.diagonal(correct_matrix, offset=distance)
                diagonal_pred = torch.diagonal(pred_matrix, offset=distance)
                # Calculate the mean accuracy
                mean_accuracy = torch.mean(diagonal_elements)
                mean_pred = torch.mean(diagonal_pred)
                # Store the mean accuracy in the dictionary
                mean_accuracies[distance] = mean_accuracy
                mean_preds[distance] = mean_pred

            metrics['accuracies'].append(mean_accuracies)
            metrics['predictions'].append(mean_preds)

            for abs_distance, accuracies in mean_accuracies.items():
                mean_accuracy = torch.mean(accuracies)
                if cfg.log.log_to_wandb:
                    wandb.log({f"mean_accuracy_distance_{abs_distance}": mean_accuracy, 'iter': n})


            # calculate the induction strength of each L2 head
            # this is the difference in attention weights from the query to the correct keys - the incorrect keys
            calc_induction_strength = False
            if calc_induction_strength:
                correct_ids = holdout_batch['label'][:, :-1] == holdout_batch['label'][:, -1].view(1, 128).T
                for h in range(cfg.model.n_heads):
                    attn_weights = out_dict_eval[f'block_1']['weights'][:, h, :, :]
                    # only get every second column, starting from the second
                    query_to_label = attn_weights[:, -1, 1::2]
                    induction_strength = query_to_label[correct_ids].mean() - query_to_label[~correct_ids].mean()
                    if cfg.log.log_to_wandb:
                        wandb.log({f'induction_strength_head_{h}': induction_strength.item(), 'iter': n})

            if holdout_accuracy == 1.:
                steps_above_criterion += 1
            else:
                steps_above_criterion = 0
            if steps_above_criterion > cfg.train.steps_above_criterion:
                print(f'holdout accuracy maximal for {steps_above_criterion} successive evaluations, stopping training')
                break

            iterdataset.set_mode('train')
            iterator = iter(dataloader)

    run.finish()
    return metrics


if __name__ == '__main__':
    import os
    from functools import partial
    from datetime import datetime

    wandb.login(key=WANDB_KEY)

    model_dir = './models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    sweep_configuration = {
        "name": "transinf-icl-sweep-{}".format(datetime.now().strftime("%Y%m%d")),
        "method": "random",
        "metric": {"goal": "minimize", "name": "loss"},
        "parameters": {
            "train.learning_rate": {"max": 1e-5, "min": 1e-7, "distribution": "uniform"},
            "train.w_decay": {"max": 1e-4, "min": 1e-7, "distribution": "uniform"},
            "model.n_blocks": {"max": 16, "min": 2, "distribution": "int_uniform"},
            "model.n_heads": {"values": [1, 2, 4, 8], "distribution": "categorical"},
            "train.warmup_steps": {"max": 10000, "min": 3000, "distribution": "int_uniform"},
            "seq.ways": {"values": [2, 3, 4, 5, 6, 7, 8], "distribution": "categorical"},
        },
        "program": "trans_inf_sweep.py"  # Add this to point to your script
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="ic_transinf_sweep")
    print(f"Sweep ID: {sweep_id}")
