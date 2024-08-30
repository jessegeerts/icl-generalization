from torch.utils.data import IterableDataset


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class MyIterableDataset(IterableDataset):
    def __init__(self, train_generator_func, holdout_generator_func, iwl_eval_generator_func):
        super(MyIterableDataset).__init__()
        self.train_generator = train_generator_func(query_distance=1)
        self.holdout_generator_func = holdout_generator_func
        self.iwl_eval_generator_func = iwl_eval_generator_func
        self.mode = 'train'
        self.eval_distance = 1
        self.holdout_generator = self.holdout_generator_func(query_distance=self.eval_distance)

    def set_mode(self, mode, eval_distance=1):
        self.mode = mode
        if mode == 'holdout':
            self.eval_distance = eval_distance
            self.holdout_generator = self.holdout_generator_func(query_distance=self.eval_distance)

    def __iter__(self):
        # print(f"Mode at start of __iter__: {self.mode}")
        if self.mode == 'train':
            # print(f"Entered 'train' branch with mode: {self.mode}")
            for item in self.train_generator():
                yield item
        elif self.mode == 'holdout':
            # print(f"Entered 'holdout' branch with mode: {self.mode}")
            for item in self.holdout_generator():
                yield item
        elif self.mode == 'iwl_eval':
            # todo: implement this
            raise NotImplementedError
        else:
            raise ValueError('Invalid mode: {}'.format(self.mode))
