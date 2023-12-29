import optuna
import traceback


class Objective:
    def __init__(self, model):
        self.model = model
        self.min_epochs = 1
        self.max_epochs = max(20, self.model.options["epochs"] * 2)

    def __call__(self, trial):
        n_layers = trial.suggest_int(
            "n_layers",
            3,
            max(15, 2 * len(self.model.options["fc_layers"])),
        )
        self.model.options["lr"] = trial.suggest_float("lr", 1e-10, 10, log=True)
        self.model.options["factor"] = trial.suggest_float("factor", 0, 1)
        self.model.options["betas"] = (
            trial.suggest_float("beta1", 0.5, 1),
            trial.suggest_float("beta2", 0.5, 1),
        )
        self.model.options["eps"] = trial.suggest_float("eps", 1e-9, 1e-7, log=True)
        self.model.options["decay"] = trial.suggest_float("decay", 1e-5, 1e-1, log=True)
        self.model.options["batch_size"] = trial.suggest_int(
            "batch_size",
            1,
            min(
                max(self.model.options["batch_size"] * 2, 1024),
                len(self.model.y) // 4,
            ),
        )
        self.model.options["accumulate"] = trial.suggest_int("accumulate", 1, 10)
        self.model.options["epochs"] = trial.suggest_int(
            "epochs", self.min_epochs, self.max_epochs
        )
        self.model.options["patience"] = trial.suggest_int(
            "patience",
            0,
            min(self.model.options["epochs"], self.model.options["patience"]),
        )
        # self.options["norm_clip"] = trial.suggest_float("norm", 1e-1, 1e1)
        if "fc_layers" in self.model.options.keys():
            self.model.options["fc_layers"] = [
                trial.suggest_int(
                    f"layer_{i}_size", len(self.model.y) // 2, len(self.model.X) * 2
                )
                for i in range(n_layers)
            ]
            self.model.options["fc_dropout"] = [
                trial.suggest_float(f"ff_dropout_{i}", 0, 1) for i in range(n_layers)
            ]
        if "RNN" in self.model.typeof:
            self.model.options["rnn_type"] = trial.suggest_categorical(
                "rnn_type", ["LSTM", "GRU"]
            )
            self.model.options["rnn_layers"] = trial.suggest_int(
                "rnn_layers",
                1,
                max(n_layers, 2 * self.model.options["rnn_layers"]),
            )
            self.model.options["rnn_dropout"] = trial.suggest_float("rnn_dropout", 0, 1)
            self.model.options["attention"] = trial.suggest_categorical(
                "attn", ["dot", "general", "concat"]
            )
            self.model.options["rnn_seq_len"] = trial.suggest_int(
                "sequence_length",
                0,
                self.model.options["batch_size"],
            )

        self.model = self.model.load(self.model.first, self.model.options)
        self.model.trial = trial
        try:
            self.model = self.model.iterate()
        except optuna.TrialPruned:
            raise
        except Exception:
            if self.model.options["verbose"] > 1:
                traceback.print_exc()
            # raise optuna.TrialPruned
            return float("inf")
        self.model = self.model.load()
        return self.model.best_loss or float("inf")

    def callback(self, study, trial):
        if study.best_trial == trial:
            self.model.trial = False
            self.model.options["tune_trials"] = 0
            self.model.save(f"{int(self.model.first.split('.')[0]) + 1}.pt")
