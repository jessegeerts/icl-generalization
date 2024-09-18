import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import random
import itertools
import matplotlib.pyplot as plt
import pandas as pd


from main_utils import log_att_weights
from sweep_utils import update_nested_config
from utils import dotdict as dd
from configs.trans_inf_config import config
from models import Transformer
from definitions import WANDB_KEY
import h5py as h5


wandb.login(key=WANDB_KEY)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------- Data -------------------------
# load the data
embeddings_file = 'datasets/omniglot_resnet18_randomized_order_s0.h5'
with h5.File(embeddings_file, 'r') as f:
    embeddings = torch.Tensor(np.array(f['resnet18/224/feat']))

# we only take the first exemplar from each class, for now
embeddings = embeddings[:, 0, :]
num_classes,  emb_dim = embeddings.shape


class TransInfSeqGen:
    def __init__(self, config):
        self.N = config.seq.N  # length of the context. For consistency between IW and IC seqs, this is same as length of ordering.
        self.n_classes = config.data.K
        # choose N classes to train on (these have a constant ordering throughout training)
        all_classes = np.arange(self.n_classes)
        np.random.shuffle(all_classes)
        self.pos_label = all_classes[0]
        self.neg_label = all_classes[1]
        self.neutral_label = all_classes[2]
        self.fixed_classes = all_classes[3:self.N+3]
        self.random_classes = all_classes[self.N+3:]

    def test_ic_train_seq(self):
        """Simple IC training sequence to see if the model can learn correct ordering.
        :return:
        """
        items = np.random.choice(self.random_classes, 3, replace=False)
        context = [
            (items[0], items[1], self.pos_label),
            (items[1], items[2], self.pos_label),
            (items[1], items[0], self.neg_label),
            (items[2], items[1], self.neg_label),
        ]
        context *= config.seq.repeats
        random.shuffle(context)
        if np.random.rand() < 0.25:
            query = (items[0], items[1])
            target = 1
        elif np.random.rand() < 0.5:
            query = (items[1], items[0])
            target = -1
        elif np.random.rand() < 0.75:
            query = (items[1], items[2])
            target = 1
        else:
            query = (items[2], items[1])
            target = -1
        return context, query, target

    def test_ic_train_seq_simplest(self):
        """Simple IC training sequence to see if the model can learn correct ordering.
        :return:
        """
        items = np.random.choice(self.random_classes, 2, replace=False)
        items = [0, 1]
        context = [
            (items[0], items[1], self.pos_label),
            (items[1], items[0], self.neg_label),
        ]
        context *= config.seq.repeats
        random.shuffle(context)
        if np.random.rand() < 0.5:
            query = (items[0], items[1])
            target = 1
        else:
            query = (items[1], items[0])
            target = -1
        return context, query, target

    def test_ic_train_seq_classify(self):
        """Simple IC training sequence to see if the model can learn correct ordering.
        :return:
        """
        items = np.random.choice(self.random_classes, 2, replace=False)
        context = [
            (items[0], self.pos_label),
            (items[1], self.neg_label),
        ]
        context *= config.seq.repeats
        random.shuffle(context)
        if np.random.rand() < 0.5:
            query = [items[0]]
            target = 1
        else:
            query = [items[1]]
            target = -1
        return context, query, target

    def get_iw_train_seq(self):
        context = self.get_random_context()
        query = self.get_fixed_query(D=1)
        target = 1
        if np.random.rand() < 0.5:
            query = (query[1], query[0])
            target = -1

        # if np.random.rand() < 0.333:
        #     query = (query[0], query[0])
        #     target = 0
        return context, query, target

    def get_ic_train_seq(self, query_type=None):
        # TODO: if this doesn't work, we can choose one symbol to mean "correct" or "delimiter" (fixed across sequences)
        # first sample N items to form the sequence
        # context contains adjacent pairs, paired up
        # query can be adjacent or distal comparison
        # sample N classes
        classes = np.random.choice(self.random_classes, self.N, replace=False)
        # create the context
        context = []
        for i in range(self.N - 1):
            context.append((classes[i], classes[i + 1]))
        # shuffle the context
        np.random.shuffle(context)
        if query_type == 'adjacent':
            query = random.choice(context)
            target = 1
            if np.random.rand() < 0.5:
                query = (query[1], query[0])
                target = -1
        else:  # choose the most distal comparison
            if np.random.rand() < 0.5:
                query = (classes[0], classes[-1])
                target = 1
            else:
                query = (classes[-1], classes[0])
                target = -1
        return context, query, target

    def get_random_context(self):
        """The context shows N-1 pairs of random examples.

        :return:
        """
        # sample N classes (randomly)
        classes = np.random.choice(self.n_classes, self.N-1, replace=False)
        classes2 = np.random.choice(self.n_classes, self.N-1, replace=False)
        context = []
        for i in range(self.N - 1):
            context.append((classes[i], classes2[i]))
        return context

    def get_fixed_query(self, D=1):
        """Generate a query pair corresponding to one of the classes with fixed ordering (to train on)
        :return:
        """
        index1 = np.random.randint(0, self.N - D)
        index2 = index1 + D
        return self.fixed_classes[index1], self.fixed_classes[index2]

    def get_iw_eval_seq(self, dist=None):
        if dist is None:
            query_distance = np.random.randint(1, self.N)
        else:
            query_distance = dist
        context = self.get_random_context()
        query = self.get_fixed_query(D=query_distance)
        target = 1
        if np.random.rand() < 0.5:
            query = (query[1], query[0])
            target = -1
        return context, query, target


# the function above gets the class indices. now we need to get the embeddings as a torch tensor, and return the
# context and query appended to each other:
def get_transitive_inference_sequence_embeddings(context, query):
    ids_seq = [item for c in context for item in c]
    ids_seq.extend([q for q in query])
    return embeddings[ids_seq]


# ------------------------- Training -------------------------
def run_experiment(config=config):

    run = wandb.init()

    sweep_params = dict(run.config)  # Get sweep parameters from wandb
    cfg = update_nested_config(config, sweep_params)  # Merge sweep params into the default config
    cfg = dd(cfg)
    print(f"Config parameters: {cfg}")

    if cfg.model.prediction_mode == 'classify':
        cfg.model.out_dim = cfg.data.L
    else:
        cfg.model.out_dim = 1  # for regression


    experiment_name = f'transitive_inference_{config.seq.train_type}_{config.model.pos_emb_type}_{config.model.pos_emb_loc}'

    metrics = {
        'iw_accuracy': [],
        'accuracies': [],
        'predictions': [],
        'loss': []
    }


    seqgen = TransInfSeqGen(config)

    # ------------------------- Model -------------------------
    model = Transformer(config=config.model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.w_decay)
    criterion = nn.MSELoss()
    losses = []
    for n in range(config.train.niters):
        model.train()
        optimizer.zero_grad()
        if config.seq.train_type == 'IC':
            context, query, target = seqgen.get_ic_train_seq('adjacent')
        elif config.seq.train_type == 'IW':
            context, query, target = seqgen.get_iw_train_seq()
        elif config.seq.train_type == 'testIC':
            context, query, target = seqgen.test_ic_train_seq_simplest()
        elif config.seq.train_type == 'testICclass':
            context, query, target = seqgen.test_ic_train_seq_classify()
        else:
            raise ValueError('Invalid training sequence type')

        if isinstance(criterion, nn.MSELoss):
            target = float(target)

        inputs = get_transitive_inference_sequence_embeddings(context, query)
        inputs = inputs.unsqueeze(0).to(device)
        target = torch.tensor([target])
        y_hat, out_dict = model(inputs, save_weights=config.save_weights)
        loss = criterion(y_hat.squeeze(0), target)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if n % config.log.logging_interval == 0:

            avg_loss = np.mean(losses)
            print(f'iteration {n}, avg loss {avg_loss}')
            if config.log.log_to_wandb:
                wandb.log({'avg_loss': avg_loss, 'iter': n})
            metrics['loss'].append(avg_loss)
            losses = []

            if config.save_weights:
                log_att_weights(n, out_dict, config)

            # ---------------------- Evaluation of adjacent and distal inferences -- in-weight ---------------------
            model.eval()

            correct_matrix = np.zeros((seqgen.N, seqgen.N))
            pred_matrix = np.zeros((seqgen.N, seqgen.N))

            ranks = np.arange(seqgen.N)
            for i, j in itertools.product(ranks, ranks):
                query = (seqgen.fixed_classes[i], seqgen.fixed_classes[j])
                context = seqgen.get_random_context()
                inputs = get_transitive_inference_sequence_embeddings(context, query)
                target = 0 if i == j else 1 if i < j else -1
                inputs = inputs.unsqueeze(0).to(device)
                y_hat, _ = model(inputs)
                pred = torch.sign(y_hat).item()
                correct = int(pred == target)

                correct_matrix[i, j] = correct
                pred_matrix[i, j] = y_hat.item()


            # Create a figure for the correct matrix
            fig_correct = plt.figure()
            plt.imshow(correct_matrix, cmap='hot', interpolation='nearest')
            plt.title('Correct Matrix')
            plt.colorbar()
            plt.close(fig_correct)  # Close the figure to prevent it from displaying in your Python environment

            # Create a figure for the pred matrix
            fig_pred = plt.figure()
            plt.imshow(pred_matrix, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)
            plt.title('Pred Matrix')
            plt.colorbar()
            plt.close(fig_pred)  # Close the figure to prevent it from displaying in your Python environment

            # Log the figures as images in wandb
            if config.log.log_to_wandb:
                wandb.log({"correct_matrix": wandb.Image(fig_correct), "pred_matrix": wandb.Image(fig_pred), 'iter': n})

            # Initialize a dictionary to store the mean accuracies for each absolute distance
            mean_accuracies = {}
            mean_preds = {}
            # Calculate the mean accuracy and output for each distance
            for distance in range(-seqgen.N+1, seqgen.N):
                # Get the elements in the diagonal at the current absolute distance
                diagonal_elements = np.diagonal(correct_matrix, offset=distance)
                diagonal_pred = np.diagonal(pred_matrix, offset=distance)
                # Calculate the mean accuracy
                mean_accuracy = np.mean(diagonal_elements)
                mean_pred = np.mean(diagonal_pred)
                # Store the mean accuracy in the dictionary
                mean_accuracies[distance] = mean_accuracy
                # store the mean prediction in the dictionary (by distance, not absolute distance)
                mean_preds[distance] = mean_pred

            metrics['accuracies'].append(mean_accuracies)
            metrics['predictions'].append(mean_preds)

            # Calculate and log the mean accuracy for each absolute distance
            for distance, accuracies in mean_accuracies.items():
                mean_accuracy = np.mean(accuracies)
                if config.log.log_to_wandb:
                    wandb.log({f"mean_accuracy_distance_{distance}": mean_accuracy, 'iter': n})

            # Calculate and log the mean prediction for each distance\
            for distance, preds in mean_preds.items():
                mean_pred = np.mean(preds)
                if config.log.log_to_wandb:
                    wandb.log({f"mean_prediction_distance_{distance}": mean_pred, 'iter': n})

    return metrics


if __name__ == '__main__':
    import os

    n_runs = 40
    all_metrics = []
    for i in range(n_runs):
        torch.manual_seed(i)
        print('-----------------------------------')
        print(f'Running experiment {i}')
        print('-----------------------------------')
        save_dir = f'results/{config.seq.train_type}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if os.path.exists(os.path.join(save_dir, f'metrics_run_{i}.csv')):
            continue

        metrics = run_experiment(config)
        all_metrics.append(metrics)

        acc_df = pd.DataFrame(metrics['accuracies']).rename(columns={d: f'mean_accuracy_at_abs(distance)_{d}' for d in metrics['accuracies'][0].keys()})
        pred_df = pd.DataFrame(metrics['predictions']).rename(columns={d: f'mean_prediction_at_distance_{d}' for d in metrics['predictions'][0].keys()})

        metrics_df = pd.concat([acc_df, pred_df], axis=1).assign(run=i)
        metrics_df.to_csv(os.path.join(save_dir, f'metrics_run_{i}.csv'), index=False)

