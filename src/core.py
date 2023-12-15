import gc
import os
import traceback
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

# ML
import optuna
import shap
import torch
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.notebook import trange, tqdm
from .data import DeepDataset
from .loss import DeepLoss

# parallelization
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split


class DeepModel(nn.Module, ABC):
    def __init__(self):
        super(DeepModel, self).__init__()
        self.trial = False
        self.labels = []
        self.predictions = []
        self.name = []
        self.fits = 0
        self.typeof = ""
        self.target_var = []
        self.hidden = None
        self.X = pd.DataFrame()
        self.y = pd.DataFrame()
        self.feature_importances_ = []

        self.apply(self.init_weights)

        options = {}
        options["timeseries"] = False
        options["lr"] = 1e-3
        options["factor"] = 0.1
        options["betas"] = (0.9, 0.999)
        options["eps"] = 1e-8
        options["decay"] = 0
        options["batch_size"] = 64
        options["epochs"] = 10
        options["patience"] = 0
        options["device"] = "cpu"
        options["use_cuda"] = False
        options["n_workers"] = 0
        options["tune_trials"] = 0
        options["accumulate"] = 1
        options["norm_clip"] = 1.0
        options["grad_scaling"] = False
        options["autocast"] = torch.float32
        options["init_scale"] = 2**16
        options["growth_factor"] = 2
        options["backoff_factor"] = 0.5
        options["growth_interval"] = 10
        options["client"] = None
        options["verbose"] = 0
        options["drop_last"] = True
        options["use_checkpoint"] = False
        self.configure(options)

    @abstractmethod
    def init_layers(self):
        # Architecture
        pass

    @abstractmethod
    def forward(self, x):
        # Forward pass through the architecture
        pass

    @abstractmethod
    def preprocess(self, X, y):
        return X, y

    @abstractmethod
    def score(self, X, y):
        pass

    @abstractmethod
    def performance(self):
        pass

    def init_weights(self, m):
        if type(m) in {nn.Linear, nn.LSTM, nn.GRU}:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.data.fill_(0.01)

    def configure(self, options):
        if not hasattr(self, "options"):
            self.options = {}
        self.options.update(options)

        self.options["accumulate"] = max(self.options["accumulate"], 1)

        self.options["use_cuda"] = (
            self.options["device"] != "cpu" and self.options["use_cuda"]
        )
        self.options["use_checkpoint"] = (
            self.options["use_checkpoint"] and self.options["use_cuda"]
        )

        # set self.typeof to the concat with space of self.typeof of all parent classes if it exists and is not empty
        self.typeof = " ".join(
            [
                parent.typeof
                for parent in self.__class__.__bases__
                if hasattr(parent, "typeof") and parent.typeof
            ]
        ).strip()

    def fit(self, X, y):
        self.X, self.y = self.preprocess(X, y)
        self.labels = []
        self.predictions = []
        self.feature_importances_ = []
        self.target_var = self.y.columns.tolist()
        if not self.name:
            self.name = self.target_var
        else:
            if self.name[-1] != self.target_var:
                self.name.append(self.target_var)

        # Condition to append "Univariate" if required
        if (
            self.fits < 1
            and len(self.target_var) == 1
            and "Univariate" not in self.typeof
        ):
            if self.typeof and self.typeof[-1] != " ":
                self.typeof += " "
            self.typeof += "Univariate"

        # Condition to append "Series"
        if self.fits >= 1 and "Series" not in self.typeof:
            if self.typeof and self.typeof[-1] != " ":
                self.typeof += " "
            self.typeof += "Series"

        # Condition to append "Parallel"
        if (
            any(isinstance(targets, list) and len(targets) > 1 for targets in self.name)
            or len(self.target_var) > 1
        ) and "Parallel" not in self.typeof:
            if self.typeof and self.typeof[-1] != " ":
                self.typeof += " "
            self.typeof += "Parallel"

        # Condition to remove "Univariate" if "Series" or "Parallel" exists
        if "Series" in self.typeof or "Parallel" in self.typeof:
            self.typeof = self.typeof.replace(" Univariate", "").strip()

        # Here, we're using test_size=0.2 in the first split to ensure that train_df is 60% of the original data,
        # valid_df is 20%, and test_df (the hold-out set) is also 20%.
        # print("Splitting into train, validation, and test sets...")
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            shuffle=not self.options["timeseries"],
            random_state=42,
        )

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            X_temp,
            y_temp,
            test_size=0.25,
            shuffle=not self.options["timeseries"],
            random_state=42,
        )

        if isinstance(self.X_train, pd.DataFrame):
            self.X_train = dd.from_pandas(self.X_train, self.options["n_workers"])
            self.y_train = dd.from_pandas(self.y_train, self.options["n_workers"])
            self.X_valid = dd.from_pandas(self.X_valid, self.options["n_workers"])
            self.y_valid = dd.from_pandas(self.y_valid, self.options["n_workers"])
            self.X_test = dd.from_pandas(self.X_test, self.options["n_workers"])
            self.y_test = dd.from_pandas(self.y_test, self.options["n_workers"])

        # reset the index of the dataframes
        self.X_train = self.X_train.reset_index(drop=True)
        self.y_train = self.y_train.reset_index(drop=True)
        self.X_valid = self.X_valid.reset_index(drop=True)
        self.y_valid = self.y_valid.reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)
        self.y_test = self.y_test.reset_index(drop=True)

        # create output directories
        output_dir = f"runs/models/{self.typeof}/{self.name}"
        outputs = ["checkpoints", "importances", "performance"]
        self.output_dirs = {}
        for output in outputs:
            self.output_dirs[output] = f"{output_dir}/{output}"
            os.makedirs(self.output_dirs[output], exist_ok=True)

        model_target = f"{'a new' if not self.fits else 'the ' + str(self.name[:-1])} {self.typeof} model to predict {self.target_var}..."
        if self.trial or self.options["tune_trials"] <= 0:
            if not self.trial and self.options["verbose"]:
                print("Training", model_target)
            self.trainer()
        else:
            if self.options["verbose"]:
                print("Tuning hyperparameters of", model_target)
            min_epochs = 1
            max_epochs = max(20, self.options["epochs"] * 2)

            def objective(trial):
                n_layers = trial.suggest_int(
                    "n_layers",
                    3,
                    max(15, 2 * len(self.options["fc_layers"])),
                )
                self.options["lr"] = trial.suggest_float("lr", 1e-10, 10, log=True)
                self.options["factor"] = trial.suggest_float("factor", 0, 1)
                self.options["betas"] = (
                    trial.suggest_float("beta1", 0.5, 1),
                    trial.suggest_float("beta2", 0.5, 1),
                )
                self.options["eps"] = trial.suggest_float("eps", 1e-9, 1e-7, log=True)
                self.options["decay"] = trial.suggest_float(
                    "decay", 1e-5, 1e-1, log=True
                )
                self.options["batch_size"] = trial.suggest_int(
                    "batch_size",
                    1,
                    min(max(self.options["batch_size"] * 2, 1024), len(self.y) // 4),
                )
                self.options["accumulate"] = trial.suggest_int("accumulate", 1, 10)
                self.options["epochs"] = trial.suggest_int(
                    "epochs", min_epochs, max_epochs
                )
                self.options["patience"] = trial.suggest_int(
                    "patience", 0, min(self.options["epochs"], self.options["patience"])
                )
                # self.options["norm_clip"] = trial.suggest_float("norm", 1e-1, 1e1)
                if "fc_layers" in self.options.keys():
                    self.options["fc_layers"] = [
                        trial.suggest_int(
                            f"layer_{i}_size", len(self.y) // 2, len(self.X) * 2
                        )
                        for i in range(n_layers)
                    ]
                    self.options["fc_dropout"] = [
                        trial.suggest_float(f"ff_dropout_{i}", 0, 1)
                        for i in range(n_layers)
                    ]
                if "RNN" in self.typeof:
                    self.options["rnn_type"] = trial.suggest_categorical(
                        f"rnn_type", ["LSTM", "GRU"]
                    )
                    self.options["rnn_layers"] = trial.suggest_int(
                        f"rnn_layers",
                        1,
                        max(n_layers, 2 * self.options["rnn_layers"]),
                    )
                    self.options["rnn_dropout"] = trial.suggest_float(
                        f"rnn_dropout", 0, 1
                    )
                    self.options["attention"] = trial.suggest_categorical(
                        f"attn", ["dot", "general", "concat"]
                    )
                    self.options["rnn_seq_len"] = trial.suggest_int(
                        f"sequence_length",
                        0,
                        self.options["batch_size"],
                    )

                self.trial = trial
                self.configure(self.options)
                try:
                    self.trainer()
                except optuna.TrialPruned:
                    raise
                except Exception:
                    if self.options["verbose"] > 1:
                        traceback.print_exc()
                    # raise optuna.TrialPruned
                    return float("inf")
                return self.best_loss or float("inf")

            # Optimize hyperparameters
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(),
                pruner=optuna.pruners.HyperbandPruner(
                    min_epochs,
                    max_epochs,
                ),
            )
            study.optimize(
                objective,
                self.options["tune_trials"],
                gc_after_trial=True,
                show_progress_bar=True,
            )  # optimize the loss
            self.trial = False
            self.options["tune_trials"] = 0

        self.fits += 1
        self.options["tune_trials"] = 0

        # save the state, with self.options, but without self.options["client"]
        torch.save(
            {
                "state_dict": self.state_dict(),
                "options": {k: v for k, v in self.options.items() if k != "client"},
            },
            "/".join([self.output_dirs["checkpoints"], "best.pt"]),
        )

        self.performance()
        self.importance()
        gc.collect()
        return self

    def partial_fit(self, X, y, train=True, i=0):
        if train:
            X.requires_grad = True
            y.requires_grad = True

        with torch.autocast(self.options["device"], self.options["autocast"]):
            pred = self(X)  # Forward pass
            loss = self.criterion(pred, y)

        if train:
            self.scaler.scale(loss / self.options["accumulate"]).backward()
            try:
                self.scaler.unscale_(self.optimizer)
            except:
                pass

            clip_grad_norm_(self.parameters(), self.options["norm_clip"])
            if i % self.options["accumulate"] == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

        return pred, loss

    def trainer(self):
        def traverse(dl, train=True):
            if train:
                self.train()
            else:
                self.eval()
            with torch.no_grad() if not train else torch.enable_grad():
                i = 0
                all_preds = []
                all_labels = []
                total_loss = 0
                self.hidden = None
                self.optimizer.zero_grad()
                for X, y in (
                    tqdm(dl) if not self.trial and self.options["verbose"] else dl
                ):
                    i += 1
                    gc.collect()
                    pred, loss = self.partial_fit(X, y, train, i)
                    total_loss += loss.item()
                    all_preds.extend(pred)
                    all_labels.extend(y)
                avg_loss = total_loss / (len(dl) or len(self.y_valid))
                if not self.trial and self.options["verbose"]:
                    print(f"Loss: {avg_loss}")
                if not train and all_labels and all_preds:
                    self.labels = torch.cat(all_labels).float().cpu().numpy()
                    # all_preds is a list of variable length tensors
                    # we need to concatenate them into a single tensor
                    # to deal with this, we first pad them with NaNs
                    # and then concatenate them
                    # Find the maximum number of rows and columns
                    max_rows = max(pred.size(0) for pred in all_preds)

                    try:
                        max_cols = max(pred.size(1) for pred in all_preds)

                        # Pad each tensor and collect them in a list
                        padded_preds = []
                        for pred in all_preds:
                            # Determine padding size for rows and columns
                            padding_rows = max_rows - pred.size(0)
                            padding_cols = max_cols - pred.size(1)

                            # Pad the tensor for rows
                            if padding_rows > 0:
                                padding_tensor_rows = torch.full(
                                    (
                                        padding_rows,
                                        pred.size(1),
                                    ),  # Match the columns of 'pred'
                                    np.nan,
                                    dtype=torch.float32,
                                ).to(pred.device)
                                pred = torch.cat([pred, padding_tensor_rows], dim=0)

                            # Pad the tensor for columns
                            if padding_cols > 0:
                                padding_tensor_cols = torch.full(
                                    (
                                        pred.size(0),
                                        padding_cols,
                                    ),  # Match the rows of 'pred' after row padding
                                    np.nan,
                                    dtype=torch.float32,
                                ).to(pred.device)
                                pred = torch.cat([pred, padding_tensor_cols], dim=1)

                            # Add to the list
                            padded_preds.append(pred)
                            all_preds = padded_preds
                    except:
                        pass

                    # Concatenate all tensors
                    self.predictions = torch.cat(all_preds, dim=0).float().cpu().numpy()
            return avg_loss

        # print("Loading layers...")
        self.init_layers()

        # save to disk:
        torch.save(
            self.state_dict(),
            "/".join([self.output_dirs["checkpoints"], "best.pt"]),
        )

        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            self.options["lr"],
            self.options["betas"],
            self.options["eps"],
            self.options["decay"],
            fused=self.options["use_cuda"],
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=self.options["factor"],
            patience=self.options["patience"],
            eps=self.options["eps"],
            verbose=not self.trial and self.options["verbose"],
        )
        self.criterion = DeepLoss(self.options["criterion"], self.options["num_out"])
        self.scaler = GradScaler(
            self.options["init_scale"],
            self.options["growth_factor"],
            self.options["backoff_factor"],
            self.options["growth_interval"],
            self.options["grad_scaling"],
        )

        self.to(self.options["device"])
        self.float()
        self.best_loss = float("inf")

        for epoch in trange(self.options["epochs"]):
            # print("Loading data...")
            self.train_ds = DeepDataset(
                self.X_train,
                self.y_train,
                self.options["rnn_seq_len"] if "RNN" in self.typeof else 0,
                self.options["batch_size"],
                self.options["device"],
                self.options["n_workers"],
                self.options["client"],
                shuffle="rnn_type" not in self.options
                and not self.options["timeseries"],
                drop_last=self.options["drop_last"],
            )

            self.valid_ds = DeepDataset(
                self.X_valid,
                self.y_valid,
                self.options["rnn_seq_len"] if "RNN" in self.typeof else 0,
                self.options["batch_size"],
                self.options["device"],
                self.options["n_workers"],
                self.options["client"],
                drop_last=self.options["drop_last"],
            )

            self.test_ds = DeepDataset(
                self.X_test,
                self.y_test,
                self.options["rnn_seq_len"] if "RNN" in self.typeof else 0,
                self.options["batch_size"],
                self.options["device"],
                self.options["n_workers"],
                self.options["client"],
                drop_last=self.options["drop_last"],
            )

            self.train_dl = DataLoader(
                self.train_ds,
                batch_size=None,
                num_workers=0
                if isinstance(self.X_train, dd.DataFrame)
                else self.options["n_workers"],
                # drop_last=True,
                # pin_memory=self.options["use_cuda"],
            )

            self.valid_dl = DataLoader(
                self.valid_ds,
                batch_size=None,
                num_workers=0
                if isinstance(self.X_valid, dd.DataFrame)
                else self.options["n_workers"],
                # drop_last=True,
                # pin_memory=self.options["use_cuda"],
            )

            self.test_dl = DataLoader(
                self.test_ds,
                batch_size=None,
                num_workers=0
                if isinstance(self.X_test, dd.DataFrame)
                else self.options["n_workers"],
                # drop_last=True,
                # pin_memory=self.options["use_cuda"],
            )

            if not self.trial and self.options["verbose"]:
                print(f"\nEpoch {epoch+1}\n--------\nTraining...")
            traverse(self.train_dl)
            if not self.trial and self.options["verbose"]:
                print("Validating...")
            avg_val_loss = traverse(self.valid_dl, False)
            if self.trial:
                self.trial.report(avg_val_loss, epoch)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()

            # If the validation losses is at a new minimum, save the model state
            if avg_val_loss < self.best_loss:
                if not self.trial and self.options["verbose"]:
                    print(
                        f"Validation loss improved ({self.best_loss} --> {avg_val_loss}).  Saving model..."
                    )
                self.best_loss = avg_val_loss
                torch.save(
                    self.state_dict(),
                    "/".join([self.output_dirs["checkpoints"], f"{epoch}.pt"]),
                )
            # if not last epoch
            if epoch != self.options["epochs"] - 1:
                self.scheduler.step(avg_val_loss)
            with torch.no_grad():
                torch.cuda.empty_cache()

        # Test the best model
        if not self.trial and self.options["verbose"]:
            print("Testing the best model...")
        self.load_state_dict(
            torch.load("/".join([self.output_dirs["checkpoints"], "best.pt"]))
        )
        traverse(self.test_dl, False)
        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()

    def predict(self, X):
        X, _ = self.preprocess(X, None)
        self.eval()
        if isinstance(X, dd.DataFrame):
            X = X.compute()
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.to_numpy().astype(np.float32)
        X = torch.from_numpy(X).to(self.options["device"])
        if "RNN" in self.typeof:
            self.hidden = None
        with torch.no_grad():
            return self(X)

    def shap(self, train, test, model, y):
        explainer = shap.DeepExplainer(model, train)
        shap_values = explainer.shap_values(test, check_additivity=False)
        feature_importances_ = shap_values

        if not isinstance(shap_values, list):
            shap_values = [shap_values]

        # Process shap values for each class
        top_shap = []
        for shap_value in shap_values:
            mean_abs_shap = np.mean(np.abs(shap_value), axis=0)
            top_indices = np.argsort(mean_abs_shap)[-min(20, len(mean_abs_shap)) :]
            top_shap.append((shap_value[:, top_indices], top_indices))

        for j, (shap_values_subset, top_indices) in enumerate(top_shap):
            plt.figure()
            plt.title(f"{y}: Important Features for Output {j}")
            shap.summary_plot(
                shap_values_subset,
                test.cpu().numpy()[:, top_indices],
                plot_type="bar",
                feature_names=np.array(self.X.columns)[top_indices],
                show=False,
            )
            plt.savefig(
                "/".join([self.output_dirs["importances"], f"shap_{y}_{j}.png"]),
                dpi=300,
            )
            if self.options["verbose"]:
                plt.show()

        return feature_importances_

    def importance(self):
        # Sample rows from the dataset as recommended by SHAP
        rg = np.random.default_rng(42)
        num_train = len(self.X_train) - 1
        indices = rg.choice(
            num_train, min(1000 if self.options["use_cuda"] else 10, num_train), False
        )
        X_train_sampled = self.train_ds.X.loc[self.X_train.index.isin(indices)]
        if isinstance(X_train_sampled, dd.DataFrame):
            X_train_sampled = X_train_sampled.compute()
        X_train_sampled = torch.tensor(
            X_train_sampled.to_numpy(), dtype=torch.float32
        ).to(self.options["device"])

        X_test = self.test_ds.X
        if isinstance(X_test, dd.DataFrame):
            X_test = X_test.compute()
        X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32).to(
            self.options["device"]
        )

        if self.options["verbose"]:
            print("Calculating SHAP values...")

        use_chkpt = self.options["use_checkpoint"]
        self.options["use_checkpoint"] = False
        self.feature_importances_ = []

        if len(self.target_var) == 1:
            self.feature_importances_.append(
                self.shap(X_train_sampled, X_test, self, self.target_var[0])
            )
        else:
            self.feature_importances_.append(
                [
                    self.shap(X_train_sampled, X_test, SingleInput(self, i), y)
                    for i, y in enumerate(self.target_var)
                ]
            )

        self.options["use_checkpoint"] = use_chkpt
        return self.feature_importances_


class SingleInput(nn.Module):
    def __init__(self, model, i):
        super(SingleInput, self).__init__()
        self.model = model
        self.i = i

    def forward(self, x):
        return self.model(x)[self.i]
