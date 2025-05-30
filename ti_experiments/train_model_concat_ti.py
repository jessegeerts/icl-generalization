import math
from functools import partial
from itertools import product
import os
import yaml

import torch
from torch import optim as optim, nn as nn
import numpy as np

import wandb
from ti_experiments.configs.cfg_ic_leave_one_out import config as default_config
from datasets.concat_ti import generate_sequences_concat_ti, generate_eval_sequences_concat_ti, \
    generate_iw_sequences_concat_ti, generate_iw_eval_sequences_concat_ti
from main_utils import log_att_weights
from models import Transformer
from utils import dotdict as dd, update_nested_config
from plotting_utils import plot_and_log_matrix
from definitions import ROOT_FOLDER

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

    checkpoint_folder = os.path.join(ROOT_FOLDER, cfg.log.checkpoint_dir, run.project, run.id)
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

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

    model = Transformer(config=cfg.model).to(device)  # my custom transformer encoder
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

    if cfg.seq.train_seq_type == 'ic':
        train_generator = partial(generate_sequences_concat_ti, batch_size=cfg.train.batch_size,
                                  item_dim=cfg.data.D // 2,
                                  leave_one_out=cfg.seq.leave_one_out)
        fixed_items = None  # not used for IC sequences
    elif cfg.seq.train_seq_type == 'iw':
        fixed_items = torch.randint(0, 2, (cfg.seq.ways, cfg.data.D // 2))
        # save fixed items to checkpoint folder for later evals
        fixed_items_path = os.path.join(checkpoint_folder, 'fixed_items.pt')
        torch.save(fixed_items, fixed_items_path)
        train_generator = partial(generate_iw_sequences_concat_ti, batch_size=cfg.train.batch_size,
                                  item_dim=cfg.data.D // 2, items=fixed_items, distractors=cfg.seq.add_distractors_for_iw_seqs)
    else:
        raise ValueError('Invalid training sequence type: {}'.format(cfg.seq.train_seq_type))

    steps_above_criterion = 0
    for n in range(cfg.train.niters):
        model.train()
        num_items = torch.randint(4, 9, (1,)).item()  # vary number of items (for IC sequences)
        batch = train_generator(num_context_items=num_items)
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        # for the transformer encoder, we need to reshape the input to (seq_len, batch_size, emb_dim)
        y_hat, _ = model(batch['example'])
        if cfg.model.prediction_mode == 'classify':
            label = batch['label'][:, -1].long()
            label[label == -1] = 0
        else:
            label = batch['label'].view(-1, 1)
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
                # log mean output value for training batch (to check for model bias)
                output_mean = y_hat.detach().mean()
                wandb.log({'output_mean_train': output_mean.item(), 'iter': n})

            # evaluate model on holdout set (same distribution as training set)
            model.eval()
            holdout_batch = train_generator(num_context_items=num_items)
            holdout_batch = {k: v.to(device) for k, v in holdout_batch.items()}

            y_hat, out_dict = model(holdout_batch['example'],
                             save_hidden_activations=cfg.save_hiddens,
                             save_weights=cfg.save_weights)
            if cfg.model.prediction_mode == 'classify':
                label = holdout_batch['label'][:, -1].long()
                label[label == -1] = 0
            else:
                label = holdout_batch['label'].view(-1, 1)

            if cfg.save_weights:
                log_att_weights(n, out_dict, cfg)

            holdout_accuracy = torch.sum(torch.sign(y_hat)==label).item() / len(y_hat)
            print(f'Holdout accuracy: {holdout_accuracy}')
            if cfg.log.log_to_wandb:
                wandb.log({'holdout_accuracy': holdout_accuracy, 'iter': n})

            # evaluate the model on all distances
            if cfg.eval_at_all_distances:
                correct_matrix, holdout_batch, pred_matrix, ranks = eval_at_all_distances(cfg, device, model, n,
                                                                                          leave_one_out=cfg.seq.leave_one_out,
                                                                                          items=fixed_items,
                                                                                          get_hiddens=False)

                plot_and_log_matrix(cfg, correct_matrix, n, ranks, ranks, 'hot', 0, 1, 'Correct Matrix')
                plot_and_log_matrix(cfg, pred_matrix, n, ranks, ranks, 'coolwarm', -1, 1, 'Pred Matrix')

            if loss < 0.0001:
                steps_above_criterion += 1
            else:
                steps_above_criterion = 0

            metrics['holdout_accuracy'].append(holdout_accuracy)
            metrics['loss'].append(loss.item())

            if steps_above_criterion > cfg.train.steps_above_criterion:
                print(f'holdout accuracy maximal for {steps_above_criterion} successive evaluations, stopping training')
                break

        if cfg.save_model and n % cfg.log.checkpoint_interval == 0:
            model_path = os.path.join(checkpoint_folder, f"model_{n}.pt")
            print(f"Saving model to {model_path}")
            torch.save(model.state_dict(), model_path)
            metrics['model_path'] = model_path
            # also save config as yaml
            config_path = os.path.join(checkpoint_folder, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(run.config, f)

    run.finish()
    return metrics


def eval_at_all_distances(cfg, device, model, n, get_hiddens=False, leave_one_out=True, items=None):
    """

    :param cfg:
    :param device:
    :param model:
    :param n: iteration number (for logging)
    :param get_hiddens:
    :param leave_one_out:
    :return:
    """
    model.eval()
    holdout_batch = None
    correct_matrix = torch.zeros((cfg.seq.ways, cfg.seq.ways))
    pred_matrix = torch.zeros((cfg.seq.ways, cfg.seq.ways))
    ranks = torch.arange(cfg.seq.ways)
    model_activations = []
    for i, j in product(ranks, ranks):
        if i == j:
            continue  # only evaluate on off-diagonal elements
        if cfg.seq.train_seq_type == 'ic':
            holdout_batch = generate_eval_sequences_concat_ti(cfg.train.batch_size, cfg.seq.ways,
                                                              cfg.data.D // 2, query_pos=(i, j),
                                                              leave_one_out=leave_one_out)
        elif cfg.seq.train_seq_type == 'iw':
            holdout_batch = generate_iw_eval_sequences_concat_ti(cfg.train.batch_size, cfg.seq.ways,
                                                                cfg.data.D // 2, items=items, query_pos=(i, j),
                                                                distractors=cfg.seq.add_distractors_for_iw_seqs)
        else:
            raise ValueError('Invalid training sequence type: {}'.format(cfg.seq.train_seq_type))
        holdout_batch = {k: v.to(device) for k, v in holdout_batch.items()}
        y_hat, out_dict = model(holdout_batch['example'], save_hidden_activations=get_hiddens)
        model_activations.append(out_dict)
        if cfg.model.prediction_mode == 'regress':
            predicted_labels = torch.sign(y_hat.squeeze())
            true_label_sign = torch.sign(holdout_batch['label'].float())
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
            wandb.log({f'output_mean_{i}_{j}': output_mean.item(), 'iter': n})
        correct_matrix[i, j] = accuracy
        pred_matrix[i, j] = output_mean
    if get_hiddens:
        return correct_matrix, holdout_batch, pred_matrix, ranks, model_activations
    else:
        return correct_matrix, holdout_batch, pred_matrix, ranks


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

    return loss, accuracy, out_dict


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    metrics = main(wandb_proj='in-context-concat')
    plt.plot(metrics['loss'])
    plt.plot(metrics['holdout_accuracy'])