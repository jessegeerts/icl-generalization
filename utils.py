from torch.utils.data import IterableDataset


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class MyIterableDataset(IterableDataset):
    def __init__(self, train_generator, holdout_generator, iwl_eval_generator):
        super(MyIterableDataset).__init__()
        self.train_generator = train_generator
        self.holdout_generator = holdout_generator
        self.iwl_eval_generator = iwl_eval_generator
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
        elif self.mode == 'iwl_eval':
            for item in self.iwl_eval_generator:
                yield item
        else:
            raise ValueError('Invalid mode: {}'.format(self.mode))
