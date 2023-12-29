import os
import torch
from .core.arch import FFN, RNN
from .core.task import Classification, Regression


class Bronte:

    archs = {
        "ffn": FFN,
        "rnn": RNN
    }

    tasks = {
        "class": Classification,
        "reg": Regression
    }

    def __init__(self, options=None, path=None, full=False):
        super(Bronte, self).__init__()
        if options is None:
            options = {}

        load = False
        if path is not None and path.endswith(".pt") and os.path.exists(path):
            checkpoint = torch.load(path)
            if not options:
                options = checkpoint["options"]
            load = True

        task = self.tasks[options["task"]] if "task" in options else FFN
        arch = self.archs[options["arch"]] if "arch" in options else Regression
        self.model = self.factory(task, arch)(options)
        if load:
            self.model = self.model.load(path, options, full=full)


    @staticmethod
    def factory(task, arch):
        class Instance(task, arch):
            def __init__(self, options={}):
                super(Instance, self).__init__()
                self.configure(options)

        return Instance

    def fit(self, X, y):
        self.model = self.model.fit(X, y)
        return 0
