import math
from functools import partial
from itertools import product
import os

import torch
from torch import optim as optim, nn as nn
from torch.utils.data import DataLoader
import numpy as np

import wandb
from configs.config_for_ic_transinf import config as default_config
from datasets.data_generators import SymbolicDatasetForSampling, TransInfSeqGenerator
from input_embedders import GaussianEmbedderForOrdering, OmniglotEmbedder
from main_utils import log_att_weights
from models import Transformer
from utils import dotdict as dd, MyIterableDataset, update_nested_config
from plotting_utils import plot_and_log_matrix
import matplotlib.pyplot as plt
from plotting_utils import TI_per_pair_plot


torch.set_num_threads(4)


def main(config=default_config, wandb_proj='ic_transinf_sweep', seed=42):

    # set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # If using CUDA

    seed_config = {'seed': seed}

    run = wandb.init(project=wandb_proj, config={**seed_config.copy(), **config.copy()})
    cfg = config.copy()

    sweep_params = dict(run.config)  # Get sweep parameters from wandb
    cfg = update_nested_config(cfg, sweep_params)  # Merge sweep params into the default config
    cfg = dd(cfg)
    for k, v in cfg.items():
        if isinstance(v, dict):
            cfg[k] = dd(v)
    print(f"Config parameters: {cfg}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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
    dataset = SymbolicDatasetForSampling(cfg.data.K)

    seqgen = TransInfSeqGenerator(dataset)

    if cfg.seq.train_seq_type == 'order':
        # todo: we can swap this for "in-weight" sequences with constant mapping
        train_generator = partial(seqgen.get_fewshot_order_seq, cfg.seq.ways, cfg.seq.shots, mode='train',
                                  train_distal=cfg.seq.include_distal_in_training)
        holdout_generator = partial(seqgen.get_fewshot_order_seq, cfg.seq.ways, cfg.seq.shots, mode='test')
        # fixme: this is just a placeholder for now
        iwl_generator = seqgen.get_fewshot_order_seq(cfg.seq.ways, cfg.seq.shots)
    elif cfg.seq.train_seq_type == 'ABBB':
        train_generator = seqgen.get_AB_BB_seqs(cfg.seq.shots)
        holdout_generator = seqgen.get_AB_BB_seqs(cfg.seq.shots)
        iwl_generator = seqgen.get_AB_BB_seqs(cfg.seq.shots)
    elif cfg.seq.train_seq_type == 'ABBA':
        train_generator = seqgen.get_AB_BA_seqs(cfg.seq.shots, set='train')
        holdout_generator = seqgen.get_AB_BA_seqs(cfg.seq.shots, set='test')
        iwl_generator = seqgen.get_AB_BA_seqs(cfg.seq.shots, set='all')
    else:
        raise ValueError('Invalid sequence type: {}'.format(cfg.seq.seq_type))

    iterdataset = MyIterableDataset(train_generator, holdout_generator, iwl_generator)
    dataloader = DataLoader(iterdataset, batch_size=cfg.train.batch_size)

    # prepare model
    if cfg.data.type == 'gaussian':
        input_embedder = GaussianEmbedderForOrdering(cfg, device)
    elif cfg.data.type == 'omniglot':
        input_embedder = OmniglotEmbedder(cfg, device)
    else:
        raise ValueError('Invalid data type: {}'.format(cfg.data.type))
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
        # batch = next(iterator)
        batch = {k: v.to(device) for k, v in next(iterator).items()}
        optimizer.zero_grad()

        # for the transformer encoder, we need to reshape the input to (seq_len, batch_size, emb_dim)
        y_hat, _ = model(batch)
        if cfg.model.prediction_mode == 'classify':
            label = batch['label'][:, -1].long()
            label[label == -1] = 0
        else:
            label = batch['label'][:, -1].float().view(-1, 1)
        loss = criterion(y_hat, label)
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
            holdout_batch = {k: v.to(device) for k, v in next(iterator).items()}
            if cfg.model.prediction_mode == 'classify':
                label = holdout_batch['label'][:, -1].long()
                label[label == -1] = 0
            else:
                label = holdout_batch['label'][:, -1].float().view(-1, 1)

            holdout_loss, holdout_accuracy, out_dict_eval, prediction = \
                eval_loss_and_accuracy(model, holdout_batch, label, criterion, cfg)
            print(f'holdout loss: {holdout_loss}, holdout accuracy: {holdout_accuracy}')
            if cfg.log.log_to_wandb:
                wandb.log({'holdout_loss': holdout_loss.item(), 'holdout_accuracy': holdout_accuracy.item(), 'output_mean': prediction.mean().item(), 'iter': n})
            metrics['holdout_accuracy'].append(holdout_accuracy.item())
            metrics['loss'].append(loss.item())
            if cfg.save_weights:
                log_att_weights(n, out_dict_eval, cfg)

            if cfg.eval_at_all_distances:
                correct_matrix, holdout_batch, pred_matrix, ranks = eval_at_all_distances(cfg, dataloader, device,
                                                                                          iterdataset,
                                                                                          model, n)

                plot_and_log_matrix(cfg, correct_matrix, n, ranks, ranks, 'hot', 0, 1, 'Correct Matrix')
                plot_and_log_matrix(cfg, pred_matrix, n, ranks, ranks, 'coolwarm', -1, 1, 'Pred Matrix')

                fig, ax = plt.subplots()
                TI_per_pair_plot(pred_matrix.cpu().numpy(), ax=ax)
                wandb.log({'TI_per_pair_plot': wandb.Image(fig), 'iter': n})
                plt.close(fig)

            # calculate the induction strength of each L2 head
            # this is the difference in attention weights from the query to the correct keys - the incorrect keys
            calc_induction_strength = False
            if calc_induction_strength:
                calculate_induction_strength(cfg, holdout_batch, n, out_dict_eval)

            if holdout_accuracy == 1.:
                steps_above_criterion += 1
            else:
                steps_above_criterion = 0
            if steps_above_criterion > cfg.train.steps_above_criterion:
                print(f'holdout accuracy maximal for {steps_above_criterion} successive evaluations, stopping training')
                break

            iterdataset.set_mode('train')
            iterator = iter(dataloader)

        if cfg.save_model and n % cfg.log.checkpoint_interval == 0:
            checkpoint_folder = os.path.join(cfg.log.checkpoint_dir, run.project, run.id)
            if not os.path.exists(checkpoint_folder):
                os.makedirs(checkpoint_folder)
            model_path = os.path.join(checkpoint_folder, f"model_{n}.pt")
            print(f"Saving model to {model_path}")
            torch.save(model.state_dict(), model_path)

    run.finish()
    return metrics


def eval_at_all_distances(cfg, dataloader, device, iterdataset, model, n, get_hiddens=False):
    holdout_batch = None
    correct_matrix = torch.zeros((cfg.seq.ways, cfg.seq.ways))
    pred_matrix = torch.zeros((cfg.seq.ways, cfg.seq.ways))
    ranks = torch.arange(cfg.seq.ways)
    model_activations = []
    for i, j in product(ranks, ranks):
        if i == j:
            continue  # only evaluate on off-diagonal elements
        iterdataset.set_mode('holdout', set_query_ranks=(i, j))
        iterator = iter(dataloader)
        model.eval()
        holdout_batch = {k: v.to(device) for k, v in next(iterator).items()}
        y_hat, out_dict = model(holdout_batch, save_hidden_activations=get_hiddens)
        model_activations.append(out_dict)
        if cfg.model.prediction_mode == 'regress':
            predicted_labels = torch.sign(y_hat.squeeze())
            true_label_sign = torch.sign(holdout_batch['label'][:, -1].float())
            accuracy = (predicted_labels == true_label_sign).float().mean()
            output_mean = y_hat.detach().mean()
        elif cfg.model.prediction_mode == 'classify':
            predicted_labels = torch.argmax(y_hat, dim=1)
            true_label = torch.where(holdout_batch['label'][:, -1] > 0, 1, 0)
            accuracy = (predicted_labels == true_label).float().mean()
            output_mean = y_hat[:, -1].detach().mean()  # mean of the "higher than" prediction
        else:
            raise ValueError('Invalid prediction mode: {}'.format(cfg.model.prediction_mode)
                             + 'Valid options are: classify, regress')

        # log the accuracy and output mean
        if cfg.log.log_to_wandb:
            wandb.log({f'accuracy_{i}_{j}': accuracy.item(), 'iter': n})
            wandb.log({f'output_mean_{i}_{j}': output_mean.item(), 'iter': n})
        correct_matrix[i, j] = accuracy
        pred_matrix[i, j] = output_mean
    if get_hiddens:
        return correct_matrix, holdout_batch, pred_matrix, ranks, model_activations
    else:
        return correct_matrix, holdout_batch, pred_matrix, ranks


def calculate_induction_strength(cfg, holdout_batch, n, out_dict_eval):
    correct_ids = holdout_batch['label'][:, :-1] == holdout_batch['label'][:, -1].view(1, 128).T
    for h in range(cfg.model.n_heads):
        attn_weights = out_dict_eval[f'block_1']['weights'][:, h, :, :]
        # only get every second column, starting from the second
        query_to_label = attn_weights[:, -1, 1::2]
        induction_strength = query_to_label[correct_ids].mean() - query_to_label[~correct_ids].mean()
        if cfg.log.log_to_wandb:
            wandb.log({f'induction_strength_head_{h}': induction_strength.item(), 'iter': n})


def eval_loss_and_accuracy(mod, inputs, labels, criterion, config):
    y_hat, out_dict = mod(inputs, save_weights=config.save_weights)

    if config.model.prediction_mode == 'regress':
        labels = labels.float()
        labels[labels == 0] = -1
    elif config.model.prediction_mode == 'classify':
        labels[labels == -1] = 0

    loss = criterion(y_hat, labels)
    if config.model.prediction_mode == 'classify':
        predicted_labels = torch.argmax(y_hat, dim=1)
    else:
        predicted_labels = torch.sign(y_hat)
    accuracy = (predicted_labels == labels).float().mean()

    return loss, accuracy, out_dict, y_hat


if __name__ == '__main__':
    main(wandb_proj='in-context-gauss-forpaper')
