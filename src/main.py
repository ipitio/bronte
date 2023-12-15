# Preprocessing / Analysis
import numpy as np
import pandas as pd
import torch
from torch import nn
from .task import ClassTask, RegTask


class Trainer(nn.Module):
    def __init__(self, options={}, path=None):
        super(Trainer, self).__init__()
        self.options = {}
        self.options["sample_size"] = 0
        self.options["verbose"] = 0
        self.options["targets"] = []
        self.options["task"] = None
        self.options["max_corr"] = 0.95
        self.options["max_clusters"] = 10
        self.options["n_workers"] = -1
        self.options["colab"] = False
        self.options["num_out"] = 1
        self.options.update(options)

        if path is not None:
            self.load(path)
        else:
            self.model = None

    @staticmethod
    def factory(task, arch):
        class Model(task, arch):
            def __init__(self, options={}):
                super(Model, self).__init__()
                self.configure(options)

        return Model

    def sample(self, data: pd.DataFrame):
        df = data.copy()
        if 0 < self.options["sample_size"] < 1:
            df = df.sample(frac=self.options["sample_size"], random_state=42)
        if 1 <= self.options["sample_size"] < len(df):
            rg = np.random.default_rng(42)
            n = rg.choice(len(df) - 1 - int(self.options["sample_size"]))
            df = df.iloc[n : n + self.options["sample_size"], :].copy()

        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError:
                    pass

        return pd.get_dummies(
            df,
            # drop_first=True,
            dtype=np.int8,
            sparse=True,
        )

    def forward(self, data: pd.DataFrame):
        assert not data.empty, "Dataframe is empty"

        if self.options["verbose"]:
            print("\nSplitting data and transforming features...")

        data = self.sample(data)

        if isinstance(self.options["targets"], str):
            self.options["target"] = [self.options["targets"]]
        X = data.drop(columns=self.options["targets"])
        y = data[self.options["targets"]]

        # if task is auto, use the number of unique values in the target to determine
        # whether to use a regression or classification model, otherwise use the task
        # specified by the user
        if self.options["task"] is None:
            # Determine the task based on the number of unique values in each column of y
            if any(
                [
                    len(y[col].unique()) > self.options["max_clusters"]
                    for col in y.columns
                ]
            ):
                self.options["task"] = RegTask
            else:
                self.options["task"] = ClassTask

        if self.model is None:
            self.model = self.factory(self.options["task"], self.options["arch"])(
                self.options
            )
        try:
            self.model = self.model.fit(X, y)
            return 0
        except Exception as e:
            if self.options["verbose"]:
                print(e)
            self.model = None
            return 1

    def load(self, path):
        checkpoint = torch.load(path)
        self.options = checkpoint["options"]
        self.model = self.factory(self.options["task"], self.options["arch"])(
            self.options
        )
        self.model.load_state_dict(checkpoint["state_dict"])
