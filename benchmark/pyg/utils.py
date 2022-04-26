import random
import numpy as np
import torch

def seed_everything(seed):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random

    Args:
        seed: the integer value seed for global random state.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, mode="max", save=False, save_path="model.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.delta = delta
        self.save = save
        if mode == "max":
            self.best_score = -np.Inf
            self.check_func = lambda x, y: x >= y
        else:
            self.best_score = np.inf
            self.check_func = lambda x, y: x <= y

    def __call__(self, score, model):
        if self.check_func(score, self.best_score + self.delta):
            if self.save:
                self.save_checkpoint(model)
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}\n")
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when score update."""
        if self.verbose:
            print(
                "Saving model ...\n"
            )
        torch.save(model.state_dict(), self.save_path)

class Logger(object):
    def __init__(self, runs, mode='max'):
        self.results = [[] for _ in range(runs)]
        if mode == 'max':
            self.arg_f = torch.argmax
            self.f = torch.max
        else:
            self.arg_f = torch.argmin
            self.f = torch.min

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = torch.tensor(self.results[run])
            
            argidx = self.arg_f(result[:, 1]).item()

            print("-" * 80)
            print(f'Run {run + 1:02d}:')
            print(f'Best Train: {self.f(result[:, 0]):.4f}')
            print(f'Best Valid: {self.f(result[:, 1]):.4f}')
            print(f'  Final Train: {result[argidx, 0]:.4f}')
            print(f'  Final Test: {result[argidx, 2]:.4f}')
            print("-" * 80)
            print()
        else:

            best_results = []
            for r in self.results:
                r = torch.tensor(r)
                train1 = self.f(r[:, 0]).item()
                valid = self.f(r[:, 1]).item()
                train2 = r[self.arg_f(r[:, 1]), 0].item()
                test = r[self.arg_f(r[:, 1]), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print("-" * 80)
            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Best Train: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, 1]
            print(f'Best Valid: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, 3]
            print(f'  Final Test: {r.mean():.4f} ± {r.std():.4f}')
            print("-" * 80)
            print()
