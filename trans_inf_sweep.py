"""
In this script I try fewshot learning with the symbolic dataset. I will use the Reddy et al. model and I will "embed"
the classes and the labels just by generating gaussian samples, much like Reddy.
"""

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import wandb

from configs.oneshot_config import config as default_config
from datasets.data_generators import SymbolicDatasetForSampling, TransInfSeqGenerator, GaussianDataset
from input_embedders import GaussianEmbedderForOrdering
from main_utils import log_att_weights
from models import Transformer
from definitions import WANDB_KEY
from utils import MyIterableDataset
from utils import dotdict as dd

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

    config = default_config.copy()
    sweep_params = {key: value for key, value in run.config.items()}
    config = dd(update_nested_config(config, sweep_params))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config.model.pos_emb_loc == 'append':
        h_dim = config.model.emb_dim + config.model.pos_dim
    else:
        h_dim = config.model.emb_dim

    experiment_name = '{}-I{}_K{}_N{}_L{}_D{}_a{}_B{}_pB{}_pC{}_eps{}_lr{}_drop{}_{}_ln{}_wDecay{}_hdim{}_{}'.format(
        config.seq.train_seq_type,
        config.train.niters,
        config.data.K,
        config.seq.N,
        config.data.L,
        config.data.D,
        config.data.alpha,
        config.seq.B,
        config.seq.pB,
        config.seq.pC,
        config.data.eps,
        config.train.learning_rate,
        config.model.drop_p,
        'custom',
        config.model.apply_ln,
        config.train.w_decay,
        h_dim,
        config.model.prediction_mode
    )
    if config.model.prediction_mode == 'classify':
        config.model.out_dim = config.data.L
    else:
        config.model.out_dim = 1  # for regression
    print(experiment_name)

    ### load or construct the dataset
    if config.data.type == 'gaussian':
        dataset = GaussianDataset(config.data.K, config.data.L, config.data.D)
    else:
        dataset = SymbolicDatasetForSampling(config.data.K)

    seqgen = TransInfSeqGenerator(dataset)
    # todo: we can swap this for "in-weight" sequences with constant mapping
    train_generator = seqgen.get_fewshot_order_seq(config.seq.ways, config.seq.shots)

    holdout_generator = seqgen.get_fewshot_order_seq(config.seq.ways, config.seq.shots)
    # fixme: this is just a placeholder for now
    iwl_generator = seqgen.get_fewshot_order_seq(config.seq.ways, config.seq.shots)
    iterdataset = MyIterableDataset(train_generator, holdout_generator, iwl_generator)
    dataloader = DataLoader(iterdataset, batch_size=config.train.batch_size)

    # prepare model
    input_embedder = GaussianEmbedderForOrdering(config)
    model = Transformer(config=config.model, input_embedder=input_embedder).to(device)  # my custom transformer encoder

    # model = MyTransformer(config, device)
    optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.w_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=.00001)


    if config.model.prediction_mode == 'classify':
        criterion = nn.CrossEntropyLoss()
    elif config.model.prediction_mode == 'regress':
        criterion = nn.MSELoss()
    else:
        raise ValueError('Invalid prediction mode: {}'.format(config.model.prediction_mode)
                         + 'Valid options are: classify, regress')

    steps_above_criterion = 0
    iterator = iter(dataloader)
    for n in range(config.train.niters):
        iterdataset.set_mode('train')
        model.train()
        batch = next(iterator)

        optimizer.zero_grad()

        # for the transformer encoder, we need to reshape the input to (seq_len, batch_size, emb_dim)
        # inputs = input_embedder(batch)
        # inputs = inputs.permute(1, 0, 2)
        y_hat, _ = model(batch)

        loss = criterion(y_hat, batch['label'][:, -1].float().view(-1, 1))
        loss.backward()
        optimizer.step()

        if n % config.log.logging_interval == 0 and n > 0:  # fixme: remove this condition
            print(f'iteration {n}, loss {loss}')
            wandb.log({'loss': loss.item(), 'iter': n})
            # log current learning rate
            for param_group in optimizer.param_groups:
                wandb.log({'lr': param_group['lr'], 'iter': n})

            # evaluate on holdout set
            iterdataset.set_mode('holdout')
            model.eval()
            holdout_batch = next(iterator)
            holdout_loss, holdout_accuracy, out_dict_eval = eval_loss_and_accuracy(model, holdout_batch, holdout_batch['label'][:, -1].long(), criterion, config)
            print(f'holdout loss: {holdout_loss}, holdout accuracy: {holdout_accuracy}')
            wandb.log({'holdout_loss': holdout_loss.item(), 'holdout_accuracy': holdout_accuracy.item(), 'iter': n})
            if config.save_weights:
                log_att_weights(n, out_dict_eval, config)

            # calculate the induction strength of each L2 head
            # this is the difference in attention weights from the query to the correct keys - the incorrect keys
            calc_induction_strength = False
            if calc_induction_strength:
                correct_ids = holdout_batch['label'][:, :-1] == holdout_batch['label'][:, -1].view(1, 128).T
                for h in range(config.model.n_heads):
                    attn_weights = out_dict_eval[f'block_1']['weights'][:, h, :, :]
                    # only get every second column, starting from the second
                    query_to_label = attn_weights[:, -1, 1::2]
                    induction_strength = query_to_label[correct_ids].mean() - query_to_label[~correct_ids].mean()
                    wandb.log({f'induction_strength_head_{h}': induction_strength.item(), 'iter': n})

            if holdout_accuracy == 1.:
                steps_above_criterion += 1
            else:
                steps_above_criterion = 0
            if steps_above_criterion == 5:
                print('holdout accuracy maximal for 5 successive evaluations, stopping training')
                break
            scheduler.step()


if __name__ == '__main__':
    import os

    wandb.login(key=WANDB_KEY)

    model_dir = './models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "loss"},
        "parameters": {
            "train.learning_rate": {"max": 0.001, "min": 0.000005, "distribution": "uniform"},
            "train.w_decay": {"max": 0.0008, "min": 0.000001, "distribution": "uniform"},
            "model.n_blocks": {"max": 8, "min": 1, "distribution": "int_uniform"},
            "model.n_heads": {"max": 8, "min": 1, "distribution": "int_uniform"},
            "model.include_mlp": {"values": [True, False], "distribution": "categorical"},
            # "seq.shots": {"max": 4, "min": 1, "distribution": "int_uniform"}
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="ic_transinf_sweep")
    wandb.agent(sweep_id=sweep_id, function=main)


