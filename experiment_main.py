"""
In this script I try fewshot learning with the symbolic dataset. I will use the Reddy et al. model and I will "embed"
the classes and the labels just by generating gaussian samples, much like Reddy.
"""

import torch
from torch.utils.data import DataLoader, IterableDataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb

from configs.fewshot_config import config
from datasets.gauss_datasets import get_mus_label_class
from datasets.partial_exposure_sequences import get_partial_exposure_sequence, exemplar_strategy
from datasets.data_generators import SymbolicDatasetForSampling, SeqGenerator
from input_embedders import GaussianEmbedder
from models import Transformer
from definitions import WANDB_KEY, ATTENTION_CMAP


class MyIterableDataset(IterableDataset):
    def __init__(self, train_generator, holdout_generator):
        super(MyIterableDataset).__init__()
        self.train_generator = train_generator
        self.holdout_generator = holdout_generator
        self.mode = 'train'

    def set_mode(self, mode):
        self.mode = mode

    def __iter__(self):
        if self.mode == 'train':
            for item in self.train_generator:
                yield item
        elif self.mode == 'holdout':
            for item in self.holdout_generator:
                yield item
        else:
            raise ValueError('Invalid mode: {}'.format(self.mode))


def eval_loss_and_accuracy(mod, inputs, labels, criterion):
    y_hat, out_dict = mod(inputs, save_weights=config.save_weights)
    loss = criterion(y_hat, labels)
    predicted_labels = torch.argmax(y_hat, dim=1)
    accuracy = (predicted_labels == labels).float().mean()
    return loss, accuracy, out_dict


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    model_dir = './models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    experiment_name = 'Fewshot-I{}_K{}_N{}_L{}_D{}_a{}_B{}_pB{}_pC{}_eps{}_lr{}_drop{}_{}_ln{}_wDecay{}_hdim{}'.format(
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
        config.model.h_dim
    )
    config.model.out_dim = config.data.L
    print(experiment_name)
    if config.log.log_to_wandb:
        wandb.login(key=WANDB_KEY)
        wandb.init(project=config.log.wandb_project, name=experiment_name, config=config)


    # prepare data
    mus_label, mus_class, labels_class = get_mus_label_class(config.data.K+3, config.data.L, config.data.D)
    mus_label = torch.Tensor(mus_label)
    mus_class = torch.Tensor(mus_class)
    labels_class = torch.Tensor(labels_class).int()

    dataset = SymbolicDatasetForSampling(config.data.S)
    seqgen = SeqGenerator(dataset,
                          config.data.n_rare_classes,
                          config.data.n_common_classes,
                          config.data.n_common_classes,
                          config.data.alpha,
                          noise_scale=0)
    train_generator = seqgen.get_fewshot_seq('zipfian', config.seq.shots, config.seq.ways, labeling='unfixed', randomly_generate_rare=False)
    holdout_generator = seqgen.get_fewshot_seq('holdout', config.seq.shots, config.seq.ways, labeling='unfixed', randomly_generate_rare=False)

    iterdataset = MyIterableDataset(train_generator, holdout_generator)
    dataloader = DataLoader(iterdataset, batch_size=config.train.batch_size)

    # prepare model
    input_embedder = GaussianEmbedder(config)
    model = Transformer(config=config.model, input_embedder=input_embedder).to(device)  # my custom transformer encoder
    optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.w_decay)
    criterion = nn.CrossEntropyLoss()

    steps_above_criterion = 0
    iterator = iter(dataloader)
    for n in range(config.train.niters):
        iterdataset.set_mode('train')
        model.train()
        batch = next(iterator)

        optimizer.zero_grad()
        # forward_pass_start = time.time()
        y_hat, out_dict = model(batch)

        loss = criterion(y_hat, batch['label'][:, -1].long())
        loss.backward()
        optimizer.step()

        if n % config.log.logging_interval == 0:
            print(f'iteration {n}, loss {loss}')
            wandb.log({'loss': loss.item(), 'iter': n})

            # evaluate on holdout set
            iterdataset.set_mode('holdout')
            model.eval()
            holdout_batch = next(iterator)
            holdout_loss, holdout_accuracy, out_dict = eval_loss_and_accuracy(model, holdout_batch, holdout_batch['label'][:, -1].long(), criterion)
            print(f'holdout loss: {holdout_loss}, holdout accuracy: {holdout_accuracy}')
            wandb.log({'holdout_loss': holdout_loss.item(), 'holdout_accuracy': holdout_accuracy.item(), 'iter': n})

            if config.save_weights:
                fig1, ax1 = plt.subplots()
                ax1.imshow(out_dict['block_0']['weights'].mean(axis=0).squeeze(), cmap=ATTENTION_CMAP)
                wandb.log({'l0_attn_map_icl': fig1,
                           'iter': n})  # note: now we're logging the mean of the attention weights across data points
                fig2, ax2 = plt.subplots()
                ax2.imshow(out_dict['block_1']['weights'].mean(axis=0).squeeze(), cmap=ATTENTION_CMAP)
                wandb.log({'l1_attn_map_icl': fig2, 'iter': n})
                plt.close('all')

            # calculate the induction strength of each L2 head
            # this is the difference in attention weights from the query to the correct keys - the incorrect keys

            correct_ids = holdout_batch['label'][:, :-1] == holdout_batch['label'][:, -1].view(1,128).T
            for h in range(config.model.n_heads):
                attn_weights = out_dict[f'block_1']['weights'][:, h, :, :]
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
    # evaluate on partial exposure paradigm

    print('Evaluating on partial exposure paradigm')
    model.eval()
    mod_preds = []
    exemplar_preds = []
    with torch.no_grad():
        for i in range(1000):
            inputs, input_names, input_labels = get_partial_exposure_sequence(config, mus_label)
            inputs = torch.Tensor(inputs).float()
            pred, out_dict = model(inputs, save_weights=True, apply_embedder=False)
            probs = torch.softmax(pred, dim=-1)
            exemplar_pred = exemplar_strategy(inputs[0, :-1:2], input_labels, inputs[0, -1, :])
            mod_preds.append(torch.argmax(pred).item())
            exemplar_preds.append(exemplar_pred)

    fig, ax = plt.subplots()
    bins = [0, 1, 2, 3]
    bin_centers = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
    ax.hist(mod_preds, bins=bins, weights=np.ones_like(exemplar_preds) / len(exemplar_preds))
    ax.set_title('Model predictions from context')
    x_tick_labels = ['B', 'X', 'none']
    ax.set_xticks(bin_centers, x_tick_labels)
    ax.set_ylim([0, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('mod_preds.png')
    if config.log.log_to_wandb:
        wandb.log({'mod_preds': fig})
        wandb.log({'model_preds': mod_preds, 'exemplar_preds': exemplar_preds})
    # save the model
    if config.save_model:
        model_path = os.path.join(model_dir, f'fewshot_pretrained_i{n}.pth')
        print(f'saving model to {model_path}')
        torch.save(model.state_dict(), model_path)


