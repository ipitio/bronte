import os
import torch
import traceback
from .arch import FFN, RNN
from .task import Classification, Regression


class Bronte:
    archs = {"ffn": FFN, "rnn": RNN}

    tasks = {"class": Classification, "reg": Regression}

    def __init__(self, options=None, path=None, full=True):
        super(Bronte, self).__init__()
        if options is None:
            options = {}

        if path is not None and path.endswith(".pt") and os.path.exists(path):
            checkpoint = torch.load(path)
            prev = checkpoint["options"]
            task = self.tasks[prev["task"]] if "task" in prev else FFN
            arch = self.archs[prev["arch"]] if "arch" in prev else Regression
            self.model = self.factory(task, arch)(prev)
            self.model = self.model.load(path, options, full=full)
        else:
            task = self.tasks[options["task"]] if "task" in options else FFN
            arch = self.archs[options["arch"]] if "arch" in options else Regression
            self.model = self.factory(task, arch)(options)

    @staticmethod
    def factory(task, arch):
        class Instance(task, arch):
            def __init__(self, options={}):
                super(Instance, self).__init__()
                self.configure(options)

        return Instance

    def fit(self, X, y):
        # try:
        self.model = self.model.fit(X, y)
        # except:
        #    if self.model.options["verbose"]:
        #        traceback.print_exc()
        #    self.model = None
        # finally:
        return 0 if self.model is not None else 1

    def predict(self, X):
        return self.model.predict(X)
