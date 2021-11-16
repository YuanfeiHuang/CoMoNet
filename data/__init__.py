from importlib import import_module
from  torch.utils.data import DataLoader

class data:
    def __init__(self, args):
        self.args = args

    def get_loader(self):
        self.module_train = import_module('data.' + self.args.data_train)
        trainset = getattr(self.module_train, self.args.data_train)(self.args)
        loader_train = DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.threads, pin_memory=False)
        return loader_train

