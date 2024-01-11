from multiprocessing import reduction
import os
import subprocess
import sys
import traceback
import warnings

verbose = 1
verbose = min(max(verbose, 0), 3)
colab = False
try:
    from google.colab import drive

    colab = True
    drive.mount("/content/drive", force_remount=True)
    drive_dir = "bronte"  # @param {type:"string"}
    path = "/".join(["drive", "MyDrive", drive_dir])
    os.chdir(path)
except:
    pass

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

cwd = os.getcwd()
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# This will install the packages below. If you use an environment manager, comment this
# out and make sure the packages in requirements.txt are installed in your environment.
if verbose:
    print("Fetching any missing dependencies...")
reqs = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
installed = [r.decode().split("==")[0] for r in reqs.split()]
failure = []
with open("requirements.txt", "r") as dependencies:
    pkgs = dependencies.readlines()
    if colab:
        pkgs.append("torch_xla\n")
    for pkg in pkgs:  # to ignore errors
        # if pkg is a comment or empty line
        if pkg.startswith("#") or len(pkg.strip()) == 0:
            continue
        pkg = pkg.strip()
        args = f"pip install {pkg} {' '.join(['-q' for _ in range(max(2, verbose) - verbose)])} --break-system-packages"
        if pkg not in installed and subprocess.call(args.split()) != 0:
            failure.append(pkg)

if len(failure) > 0 and failure[0]:
    print("Failed to install the following packages: " + str(failure))
    print("Try installing them manually.")
    raise SystemExit(1)
elif verbose:
    print("All dependencies are installed.")

warnings.filterwarnings("ignore")
os.chdir(cwd)

import torch
import numpy as np
from torch import nn
from .arch import FFN, RNN
from .task import Classification, Regression

# set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class Bronte:
    archs = {"ffn": FFN, "rnn": RNN}

    tasks = {"class": Classification, "reg": Regression}

    losses = {
        "mse": nn.MSELoss,
        "cross": nn.CrossEntropyLoss,
        "bce": nn.BCELoss,
        "bcew": nn.BCEWithLogitsLoss,
        "nll": nn.NLLLoss,
        "huber": nn.HuberLoss,
    }

    def __init__(self, options=None, path=None, full=True):
        super().__init__()
        if options is None:
            options = {}

        options["verbose"] = verbose  # flag
        options["colab"] = colab  # flag

        if path is not None and path.endswith(".pt") and os.path.exists(path):
            checkpoint = torch.load(path)
            prev = checkpoint["options"]
            task = (
                self.tasks[options["task"]]
                if "task" in options and options["task"]
                else self.tasks[prev["task"]]
            )
            arch = (
                self.archs[options["arch"]]
                if "arch" in options and options["arch"]
                else self.archs[prev["arch"]]
            )
            options["criterion"] = (
                self.losses[options["criterion"]](reduction="none")
                if "criterion" in options
                else self.losses[prev["criterion"]]
            )
            self.model = self.factory(task, arch)(prev)
            self.model = self.model.load(path, options, full=full)
        else:
            task = (
                self.tasks[options["task"]]
                if "task" in options and options["task"]
                else FFN
            )
            arch = (
                self.archs[options["arch"]]
                if "arch" in options and options["arch"]
                else Regression
            )
            options["criterion"] = (
                self.losses[options["criterion"]](reduction="none")
                if "criterion" in options
                else nn.MSELoss(reduction="none")
            )
            self.model = self.factory(task, arch)(options)

    @staticmethod
    def factory(task, arch):
        class Instance(task, arch):
            def __init__(self, options={}):
                super().__init__()
                self.configure(options)

        return Instance

    def fit(self, data):
        X = data.drop(columns=self.model.options["targets"])
        y = data[self.model.options["targets"]]
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
