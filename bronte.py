import os
import subprocess
import torch
import traceback
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, pool
from torch import nn
from src.arch import FFN, RNN
from src.task import Classification, Regression
try:
    from google.colab import drive
    colab = True
except:
    colab = False
verbose = 1 # 0-3

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
                else prev["criterion"]
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
        df = data.copy()
        options = self.model.options
        if 1 <= options["sample_size"] < len(df) or options["timeseries"]:
            if options["sample_size"] >= len(df):
                options["sample_size"] = len(df) - 1
            if options["sample_size"] < 1:
                options["sample_size"] = np.rint(options["sample_size"] * len(df))
            rg = np.random.default_rng(42)
            n = rg.choice(len(df) - 1 - np.rint(options["sample_size"]))
            df = df.iloc[n : n + options["sample_size"], :].copy()
        elif 0 < options["sample_size"] < 1:
            df = df.sample(frac=options["sample_size"], random_state=42)
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError:
                    pass

        data = pd.get_dummies(df, dtype=np.int8)
        if not options["timeseries"]:
            X = data.drop(columns=options["targets"])
            y = data[options["targets"]]
        else:
            X = data.iloc[: -options["num_out"], :]
            y = data.iloc[-options["num_out"] :, :][options["targets"]]
        # try:
        self.model = self.model.fit(X, y)
        # except:
        #    if options["verbose"]:
        #        traceback.print_exc()
        #    self.model = None
        # finally:
        return 0 if self.model is not None else 1

    def predict(self, X):
        return self.model.predict(X)


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

if colab:
    drive.mount("/content/drive", force_remount=True)
    drive_dir = "bronte"  # @param {type:"string"}
    path = "/".join(["drive", "MyDrive", drive_dir])
    os.chdir(path)
else:
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

# set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

app = Flask(__name__)

engine = create_engine(
    "sqlite:///data/data.db",
    connect_args={"check_same_thread": False},
    poolclass=pool.SingletonThreadPool,
    echo=verbose > 2,
)
prefix = "bronte_"
idx = 0

tensorboard = None

@app.route("/flush", methods=["GET"])
def flush():
    cursor = engine.raw_connection().cursor()
    for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table';"):
        cursor.execute(f"DROP TABLE {row[0]}")
    try:
        return "Database flushed"
    except:
        pass


@app.route("/load", methods=["POST"])
def load(data=None):
    if data is None:
        data = request.get_json()["data"]
    df = pd.read_json(data)
    global idx
    df.to_sql(f"{prefix}{idx}", engine, if_exists="replace", index=False)
    idx += 1
    try:
        return jsonify({"message": f"Data loaded into table {prefix}{idx}"}), 200
    except:
        pass


@app.route("/fit", methods=["POST"])
def fit(models=None):
    if models is None:
        models = request.get_json()

    trainers = []
    for task_type in models["tasks"]:
        for task in models["tasks"][task_type]:
            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table';", engine
            )
            for arch in models["archs"]:
                options = task | arch
                path = ""
                if "fine" in task_type:
                    path = (
                        f"models/{options['task']} {options['arch']}/{options['prev']}/"
                    )
                    path += f"{sorted(os.listdir(path))[-1]}/model.pt"

                trainer = Bronte(options, path)
                for i, table in enumerate(tables["name"].tolist()):
                    if options["table_prefix"] in table:
                        data = pd.read_sql(f"SELECT * FROM {table}", engine)
                        if trainer.fit(data) != 0 or i >= options["samples"] - 1:
                            break
                trainers.append(trainer)
    try:
        return jsonify({"message": "Models trained successfully"}), 200
    except:
        return trainers


@app.route("/logs", methods=["GET"])
def track():
    global tensorboard
    if tensorboard is not None:
        tensorboard.terminate()
        tensorboard = None
        try:
            return jsonify({"message": "Tensorboard server stopped"}), 200
        except:
            pass
    if not colab:
        tensorboard = subprocess.Popen(["tensorboard", "--logdir=models", "--bind_all"])
        try:
            return jsonify({"message": "Tensorboard server started"}), 200
        except:
            pass


@app.route("/predict", methods=["POST"])
def predict(XX=None, paths=[]):
    if XX is None:
        try:
            XX = request.get_json()  # Get new data from the request
        except:
            pass
    if not isinstance(XX, list):
        XX = [XX]
    if not paths:
        for root, _, files in os.walk("models"):
            for file in files:
                if (
                    file.endswith(".pt")
                    and not file.split(".")[0].lstrip("-").isdigit()
                ):
                    paths.append(os.path.join(root, file))
    predictions = {}
    for path in paths:
        model = Bronte(path=path).model
        targets = model.target_var
        for i, X in enumerate(XX):
            if verbose:
                print(f"Predicting {targets} with {path}...\n")
            preds = model.predict(X)
            if not isinstance(preds, list):
                preds = [preds.squeeze()]
            predictions[path] = {str(i): preds}
            if verbose:
                for pred, target in zip(preds, targets):
                    print(f"Head of {target} predictions:\n{pred.squeeze()[:10]}\n")

    try:
        return jsonify(predictions), 200
    except:
        return predictions


if __name__ == "__main__":
    app.run(debug=True)
