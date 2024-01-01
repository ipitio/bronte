import gc
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
from abc import ABC, abstractmethod
from copy import deepcopy

# ML
import optuna
import shap
import torch
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.notebook import trange, tqdm
from .data import DeepDataset
from .loss import DeepLoss
from .tune import Objective

# parallelization
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split


class Model(nn.Module, ABC):
    def __init__(self):
        super(Model, self).__init__()
        self.options = {}
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
        self.epoch = 0
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None
        self.best_loss = float("inf")
        self.X_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.X_valid = pd.DataFrame()
        self.y_valid = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_test = pd.DataFrame()
        self.train_ds = None
        self.valid_ds = None
        self.test_ds = None
        self.train_dl = None
        self.valid_dl = None
        self.test_dl = None
        self.action = "Training"
        self.output_dir = "models"
        self.output_dirs = {}
        self.typeof = ""
        self.studies = "/".join([self.output_dir, "tuning.db"])
        self.writer = None

        # task
        self.options["task"] = ""
        self.options["sample_size"] = 0
        self.options["verbose"] = 0
        self.options["targets"] = []
        self.options["max_corr"] = 0.95
        self.options["max_clusters"] = 10
        self.options["n_workers"] = -1
        self.options["colab"] = False
        self.options["num_out"] = 1

        # arch
        self.options["arch"] = ""
        self.options["timeseries"] = False
        self.options["lr"] = 1e-3
        self.options["factor"] = 0.1
        self.options["betas"] = (0.9, 0.999)
        self.options["eps"] = 1e-8
        self.options["decay"] = 0
        self.options["batch_size"] = 64
        self.options["epochs"] = 10
        self.options["patience"] = 0
        self.options["device"] = "cpu"
        self.options["use_cuda"] = False
        self.options["n_workers"] = 0
        self.options["tune_trials"] = 0
        self.options["accumulate"] = 1
        self.options["norm_clip"] = 1.0
        self.options["grad_scaling"] = False
        self.options["autocast"] = torch.float32
        self.options["init_scale"] = 2**16
        self.options["growth_factor"] = 2
        self.options["backoff_factor"] = 0.5
        self.options["growth_interval"] = 10
        self.options["client"] = None
        self.options["verbose"] = 0
        self.options["drop_last"] = True
        self.options["use_checkpoint"] = False
        self.options["resume"] = True
        self.options["restart"] = False

    @abstractmethod
    def preprocess(self, X, y):
        return X, y

    @abstractmethod
    def score(self, X, y):
        pass

    @abstractmethod
    def performance(self):
        pass

    @abstractmethod
    def init_layers(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    def init_weights(self, m):
        try:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.data.fill_(0.01)
        except:
            pass

    def configure(self, options={}):
        self.options.update(options)

        self.options["accumulate"] = max(self.options["accumulate"], 1)

        self.options["device"] = (
            "cuda"
            if self.options["use_cuda"]
            else "cpu"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.options["use_cuda"] = (
            self.options["device"] != "cpu" and self.options["use_cuda"]
        )
        self.options["use_checkpoint"] = (
            self.options["use_checkpoint"] and self.options["use_cuda"]
        )
        self.options["grad_scaling"] = (
            self.options["grad_scaling"] and self.options["use_cuda"]
        )
        self.options["autocast"] = (
            torch.float16 if self.options["use_cuda"] else torch.bfloat16
        )

        self.options["n_workers"] = (
            self.options["n_workers"]
            if self.options["n_workers"] >= 0
            else os.cpu_count()
        )

        if self.options["task"] and self.options["arch"]:
            self.typeof = " ".join([self.options["task"], self.options["arch"]])

        return self

    def load(
        self,
        path="best",
        options={},
        state={
            "model": None,
            "optimizer": None,
            "scheduler": None,
            "scaler": None,
            "instance": {},
        },
        full=False,
    ):
        def set_attr(obj, names, val):
            return (
                setattr(obj, names[0], val)
                if len(names) == 1
                else set_attr(getattr(obj, names[0]), names[1:], val)
            )

        found = True
        if path == "best":
            epochs = []
            for file in os.listdir(self.output_dirs["checkpoints"]):
                if file.endswith(".pt") and file.split(".")[0].lstrip("-").isdigit():
                    epochs.append(int(file.split(".")[0]))
            while epochs:
                if self.load(f"{max(epochs)}.pt") is not None:
                    self.epoch = max(max(epochs) + 1, self.epoch)
                    return self
                epochs.remove(max(epochs))
            found = False
        elif path:
            if not full:
                path = "/".join([self.output_dirs["checkpoints"], path])
            checkpoint = torch.load(path)
            state = checkpoint["state"]
            if not options:
                options = checkpoint["options"]

        self.configure(options)
        self.__dict__.update(state["instance"])

        self.init_layers()
        self.apply(self.init_weights)
        if state["model"] is None:
            state["model"] = deepcopy(self.state_dict())
        size_mismatched = {}
        old = deepcopy(state["model"])
        for k, v in old.items():
            if k not in self.state_dict() or v.size() != self.state_dict()[k].size():
                if k in self.state_dict():
                    size_mismatched[k] = state["model"][k]
                del state["model"][k]
        self.load_state_dict(state["model"], False)
        for k, v in size_mismatched.items():
            try:
                set_attr(
                    self,
                    k.split("."),
                    torch.nn.Parameter(
                        v.resize_(self.state_dict()[k].size()), v.requires_grad
                    ).to(self.options["device"]),
                )
            except:
                pass

        self.criterion = DeepLoss(self.options["criterion"], self.options["num_out"])
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
        self.scaler = GradScaler(
            self.options["init_scale"],
            self.options["growth_factor"],
            self.options["backoff_factor"],
            self.options["growth_interval"],
            self.options["grad_scaling"],
        )
        try:
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.scaler.load_state_dict(state["scaler"])
        except:
            pass
        return self if found else None

    def save(self, path=None, full=False):
        if path is None:
            path = f"{self.epoch - 1}.pt"
        if not full:
            path = "/".join([self.output_dirs["checkpoints"], path])
        state = {
            "model": deepcopy(self.state_dict()),
            "optimizer": deepcopy(self.optimizer.state_dict())
            if self.optimizer is not None
            else None,
            "scheduler": deepcopy(self.scheduler.state_dict())
            if self.scheduler is not None
            else None,
            "scaler": deepcopy(self.scaler.state_dict())
            if self.scaler is not None
            else None,
            "instance": {
                key: value
                for key, value in self.__dict__.items()
                # and key none of "options" or "writer"
                if not key.startswith("__")
                and not callable(key)
                and key
                not in [
                    "options",
                    "writer",
                ]
            },
        }
        torch.save(
            {
                "options": {k: v for k, v in self.options.items() if k != "client"},
                "state": state,
            },
            path,
        )
        return state

    def restart(self):
        # delete trials
        if os.path.exists(self.studies):
            os.remove(self.studies)

        # remove everything greater than first
        for file in os.listdir(self.output_dirs["checkpoints"]):
            if file.endswith(".pt") and file.split(".")[0].lstrip("-").isdigit():
                os.remove(os.path.join(self.output_dirs["checkpoints"], file))
        return self

    def resume(self, name, restart=False):
        if len(self.name) < 1 or self.name[-1] != self.target_var:
            self.name.append(name)

        outputs = ["checkpoints", "importances", "performance"]
        for output in outputs:
            self.output_dirs[output] = (
                f"{self.output_dir}/{output}"
                if output == "checkpoints"
                else f"{self.output_dir}/{self.typeof}/{self.name}/{self.fits}/{output}"
            )
            os.makedirs(self.output_dirs[output], exist_ok=True)

        for root, dirs, _ in os.walk(self.output_dir):
            for folder in dirs:
                if folder == "[[]]":
                    shutil.rmtree(f"{root}/{folder}", True)

        if len(self.name) > 1 and self.name[-1] != self.target_var:  # transfer
            self.epoch = 0
            self.best_loss = float("inf")
            self.fits = 0
            self.action = "Fine-tuning"
            if self.options["freeze"]:
                for name, param in self.named_parameters():
                    if name in self.options["freeze"]:
                        param.requires_grad = False
            if restart:
                self.restart()

    def partial_fit(self, X, y, train=True, i=0):
        if train:
            X.requires_grad = True
            y.requires_grad = True

        with torch.autocast(self.options["device"], self.options["autocast"]):
            pred = self(X)  # Forward pass
            loss = self.criterion.mean(pred, y)

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

    def partial_iterate(self, dl, train=False):
        if train:
            self.train()
        with torch.no_grad() if not train else torch.enable_grad():
            i = 0
            all_preds = []
            all_labels = []
            total_loss = 0
            self.hidden = None
            self.optimizer.zero_grad()
            for X, y in tqdm(dl) if not self.trial and self.options["verbose"] else dl:
                i += 1
                gc.collect()
                pred, loss = self.partial_fit(X, y, train, i)
                total_loss += loss.item()
                if dl is self.test_dl:
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
        self.eval()
        return avg_loss

    def iterate(self):
        self.to(self.options["device"])
        self.float()
        end = self.options["epochs"] * (self.fits + 1)
        skipped = (
            0 if self.fits == 0 else self.options["epochs"] * self.fits - self.epoch
        )

        saved = False
        for _ in trange(end - skipped - self.epoch):
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

            if not self.trial and self.options["verbose"]:
                print(f"\nEpoch {self.epoch+1}\n--------\nTraining...")
            avg_train_loss = self.partial_iterate(self.train_dl, True)
            if self.writer is not None:
                self.writer.add_scalar("Loss/Train", avg_train_loss, self.epoch)
            if not self.trial and self.options["verbose"]:
                print("Validating...")
            avg_val_loss = self.partial_iterate(self.valid_dl)
            if self.writer is not None:
                self.writer.add_scalar("Loss/Valid", avg_val_loss, self.epoch)
            if self.trial:
                self.trial.report(avg_val_loss, self.epoch)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()
            self.scheduler.step(avg_val_loss, self.epoch)
            self.epoch += 1
            # If the validation losses is at a new minimum, save the model state
            if avg_val_loss < self.best_loss:
                if not self.trial and self.options["verbose"]:
                    print(f"Loss decreased; saving model...")
                self.best_loss = avg_val_loss
                self.save()
                saved = True
            with torch.no_grad():
                torch.cuda.empty_cache()

        # Test the best model
        if saved:
            if not self.trial and self.options["verbose"]:
                print(f"\nBest: Epoch {self.epoch - 1}\n--------\nTesting...")

            if self.load() is None:
                raise Exception("No trained model found")

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

            self.test_dl = DataLoader(
                self.test_ds,
                batch_size=None,
                num_workers=0
                if isinstance(self.X_test, dd.DataFrame)
                else self.options["n_workers"],
                # drop_last=True,
                # pin_memory=self.options["use_cuda"],
            )
            self.save()

        avg_test_loss = self.partial_iterate(self.test_dl)
        if self.writer is not None:
            self.writer.add_scalar("Loss/Test", avg_test_loss, self.epoch - 1)
        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()
        return self

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
        X_train_sampled = self.X_train.loc[self.X_train.index.isin(indices)]
        if isinstance(X_train_sampled, dd.DataFrame):
            X_train_sampled = X_train_sampled.compute()
        X_train_sampled = torch.tensor(
            X_train_sampled.to_numpy(), dtype=torch.float32
        ).to(self.options["device"])

        X_test = self.X_test
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
            for i, y in enumerate(self.target_var):
                self.feature_importances_.append(
                    self.shap(X_train_sampled, X_test, SingleInput(self, i), y)
                )

        self.options["use_checkpoint"] = use_chkpt
        return self.feature_importances_

    def fit(self, X, y):
        resumed = ""
        if self.options["resume"] or self.options["restart"]:
            self.resume(y.columns.tolist())
            if self.options["restart"] or self.load() is None:
                if self.options["restart"]:
                    self.restart()
                for k, v in self.output_dirs.items():
                    if k != "checkpoints":
                        shutil.rmtree(v, True)
                self.name = []
            elif self.options["resume"]:
                if self.options["epochs"] * (self.fits + 1) - self.epoch <= 0:
                    if self.options["verbose"]:
                        print("Model has already been trained")
                    return self
                resumed = "Resuming "
            if self.options["resume"]:
                self.options["restart"] = False

        self.X, self.y = self.preprocess(X, y)
        self.labels = []
        self.predictions = []
        self.feature_importances_ = []
        self.target_var = self.y.columns.tolist()
        self.resume(self.target_var, True)

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

        # If the seed is set, the same sequence of rows will be selected each time.
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

        self.load("")
        self.writer = SummaryWriter(
            f"models/{self.typeof}/{self.name}/{self.fits}/runs"
        )

        model_target = f"{'a new' if not self.fits else 'the ' + str(self.name[:-1]) if len(self.name) > 1 else str(self.name[0])} {self.typeof} model to predict {self.target_var}..."
        if self.trial or self.options["tune_trials"] <= 0:
            if not self.trial and self.options["verbose"]:
                print(f"{resumed}{self.action} {model_target}")
            self.iterate()
        else:
            self.save("-1.pt")
            if self.options["verbose"]:
                print(f"{resumed}Tuning {self.action} {model_target}")
            objective = Objective(self)
            study = optuna.create_study(
                storage=f"sqlite:///{self.studies}",
                sampler=optuna.samplers.TPESampler(),
                pruner=optuna.pruners.HyperbandPruner(
                    objective.min_epochs,
                    objective.max_epochs,
                ),
                study_name=f"{self.typeof} {self.name} {self.fits}.{self.epoch}",
                direction="minimize",
                load_if_exists=self.options["resume"],
            )
            study.optimize(
                objective,
                self.options["tune_trials"],
                gc_after_trial=True,
                show_progress_bar=True,
            )  # optimize the loss
            if self.load("-2.pt") is None:
                raise Exception("No tuned model found")
            self.trial = False
            self.options["tune_trials"] = 0

        self.fits += 1
        self.performance()
        self.importance()
        self.save()
        self.save(
            f"{self.output_dir}/{self.typeof}/{self.name}/{self.fits - 1}/model.pt",
            True,
        )

        self.writer.flush()
        self.writer.close()

        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()
        return self

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


class SingleInput(nn.Module):
    def __init__(self, model, i):
        super(SingleInput, self).__init__()
        self.model = model
        self.i = i

    def forward(self, x):
        return self.model(x)[self.i]
