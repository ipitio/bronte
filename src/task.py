import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    QuantileTransformer,
    RobustScaler,
    KBinsDiscretizer,
    LabelEncoder,
)
from torch import nn
from torchmetrics.functional.classification import (
    accuracy,
    auroc,
    f1_score,
    precision,
    recall,
    confusion_matrix,
)
from torchmetrics.functional.regression import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from .trainer.base import Model


class Classification(Model):
    def __init__(self):
        super().__init__()
        self.options["criterion"] = nn.MultiLabelMarginLoss(reduction="none")

    def bins(self, data):
        # Reshape data either using array.reshape(-1, 1) if data has a single feature or array.reshape(1, -1) if it contains a single sample.
        data = data.values.reshape(-1, 1)

        # estimate using Silhouette Analysis
        silhouette_scores = []
        for k in range(2, self.options["max_clusters"] + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init="auto", algorithm="auto")
            labels = km.fit_predict(data)
            silhouette_avg = silhouette_score(data, labels)
            silhouette_scores.append(silhouette_avg)

        return np.argmax(silhouette_scores) + 2  # Adding 2 because range starts at 2

    def preprocess(self, X, y):
        if y is not None:
            for col in y.columns:
                if y[col].dtype == "object":
                    # Encode categorical targets
                    le = LabelEncoder()
                    y[col] = le.fit_transform(y[col].values)
                else:
                    optimal_n = self.bins(y[col])
                    if y[col].dtype == "float" or (
                        y[col].dtype in ["int", "int32", "int64"]
                        and len(y[col].unique()) > optimal_n
                    ):
                        # Only bin numerical targets that are not already discrete or have too many unique values
                        kbins = KBinsDiscretizer(
                            n_bins=optimal_n, encode="ordinal", strategy="quantile"
                        )
                        y[col] = (
                            kbins.fit_transform(y[col].values.reshape(-1, 1))
                            .astype(int)
                            .flatten()
                        )

            # make self.options["num_out"] a list where each element is the number of unique values in each column of y
            self.options["num_out"] = [len(y[col].unique()) for col in y.columns]

        # Remove highly correlated features
        corr = np.corrcoef(X.values, rowvar=False)
        corr_matrix = np.abs(corr)
        r, c = np.triu_indices(corr_matrix.shape[0], k=1)
        to_drop = set()
        for i, j in zip(r, c):
            if corr_matrix[i, j] > self.options["max_corr"]:
                to_drop.add(j)
        X.drop(columns=X.columns[list(to_drop)], inplace=True)

        return X, y

    def score(self, X, y):
        return [
            accuracy(
                self.predict(X), y, "multiclass", num_classes=self.options["num_out"][i]
            ).item()
            for i, _ in enumerate(self.y.columns)
        ].mean()

    def performance(self):
        if self is not None:
            self.accuracy = []
            self.f1 = []
            self.precision = []
            self.recall = []
            self.auroc = []
            for i, col in enumerate(self.y.columns):
                if len(self.y.columns) > 1:
                    index = self.y.columns.get_loc(col)
                    y = self.labels[:, index] if self.labels.ndim > 1 else self.labels
                    preds = (
                        self.predictions[:, index]
                        if self.predictions.ndim > 1
                        else self.predictions
                    )
                else:
                    y = self.labels.flatten() if self.labels.ndim > 1 else self.labels
                    preds = (
                        self.predictions.flatten()
                        if self.predictions.ndim > 1
                        else self.predictions
                    )

                if len(preds) == 0 or len(y) == 0:
                    return

                # scale pred to number of classes and round to nearest integer
                preds = np.rint(preds * (self.options["num_out"][i] - 1))

                length = min(len(y), len(preds))
                preds = torch.tensor(
                    preds[:length].squeeze(), device=self.options["device"]
                ).cpu()
                y = torch.tensor(y[:length], device=self.options["device"]).cpu()

                num_classes = self.options["num_out"][i]
                task = "multiclass"

                # preds and y have to be non-negative
                if preds.min() < 0 or y.min() < 0:
                    smallest = min(preds.min(), y.min())
                    preds -= smallest
                    y -= smallest

                # preds and y have to be integers
                preds = preds.to(torch.int64)
                y = y.to(torch.int64)

                accuracy_score = accuracy(preds, y, task, num_classes=num_classes)
                f1 = f1_score(preds, y, task, average="macro", num_classes=num_classes)
                precision_score = precision(
                    preds, y, task, average="macro", num_classes=num_classes
                )
                recall_score = recall(
                    preds, y, task, average="macro", num_classes=num_classes
                )
                cm = confusion_matrix(preds, y, task, num_classes=num_classes)

                # preds should be one more dimension than y
                if preds.ndim < y.ndim + 1:
                    preds = preds.unsqueeze(y.ndim + 1 - preds.ndim)
                if preds.ndim > y.ndim + 1:
                    preds = preds.squeeze(preds.ndim - y.ndim - 1)

                # preds.shape[1] should be equal to num_classes
                if preds.shape[1] < num_classes:
                    preds = torch.cat(
                        [
                            preds,
                            torch.zeros(
                                preds.shape[0], num_classes - preds.shape[1]
                            ).to(preds.device),
                        ],
                        dim=1,
                    ).to(preds.device)
                if preds.shape[1] > num_classes:
                    preds = preds[:, :num_classes]

                auroc_score = auroc(
                    preds, y.to(torch.int64), task, num_classes=num_classes
                )

                if self.options["verbose"]:
                    print(
                        f"\nPerformance of {self.typeof} model for {col}:\n"
                        f"\tAccuracy: {accuracy_score}\n"
                        f"\tF1 Score: {f1}\n"
                        f"\tPrecision: {precision_score}\n"
                        f"\tRecall: {recall_score}\n"
                        f"\tArea Under ROC Curve: {auroc_score}"
                    )

                # Plot the confusion matrix
                plt.figure(figsize=(8, 8))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.title(f"Confusion Matrix of {self.name} for {col}")
                plt.xlabel("Predicted Labels")
                plt.ylabel("True Labels")
                plt.savefig(
                    "/".join([self.output_dirs["performance"], f"cm_{col}.png"])
                )
                if self.options["verbose"]:
                    plt.show()

                self.accuracy.append(float(accuracy_score))
                self.f1.append(float(f1))
                self.precision.append(float(precision_score))
                self.recall.append(float(recall_score))
                self.auroc.append(float(auroc_score) if auroc_score else 0)


class Regression(Model):
    def __init__(self):
        super().__init__()
        self.options["criterion"] = nn.HuberLoss(reduction="none")

    def preprocess(self, X, y):
        if y is not None:
            self.options["num_out"] = [1] * len(y.columns)

            clf = IsolationForest(random_state=42, n_jobs=-1, warm_start=True)
            clf.fit(y)
            mask = clf.predict(y) == 1
            y = y[mask]
            X = X[mask]

        for lim in np.arange(self.options["max_corr"], 0, -0.01):
            # Remove highly correlated features
            corr = np.corrcoef(X.values, rowvar=False)
            corr_matrix = np.abs(corr)
            r, c = np.triu_indices(corr_matrix.shape[0], k=1)
            to_drop = set()
            for i, j in zip(r, c):
                if corr_matrix[i, j] > lim:
                    to_drop.add(j)
            X_dropped = X.drop(columns=X.columns[list(to_drop)])

            # Define the transformations pipeline
            transformers = Pipeline(
                [
                    (
                        "normality",
                        QuantileTransformer(
                            output_distribution="normal",
                            random_state=42,
                            copy=False,
                        ),
                    ),
                    ("variance", RobustScaler(unit_variance=True, copy=False)),
                ]
            )
            # Apply the transformations on X
            try:
                X_transformed = transformers.fit_transform(X)
                X = pd.DataFrame(X_transformed, columns=X.columns)
                break
            except np.linalg.LinAlgError:
                pass

        return X, y

    def score(self, X, y):
        self.r2 = r2_score(self.predict(X), y)
        return self.r2

    def performance(self):
        if self is not None:
            length = min(len(self.labels), len(self.predictions))
            preds = (
                torch.tensor(
                    self.predictions[:length], device=self.options["device"]
                ).cpu()
            ).squeeze()
            y = torch.tensor(self.labels[:length], device=self.options["device"]).cpu()
            if preds.ndim == 0 or y.ndim == 0:
                return
            try:
                rmse = mean_squared_error(preds, y, False, len(self.target_var))
                mae = mean_absolute_error(preds, y)
                r2 = r2_score(preds, y)
                corr = np.corrcoef(preds, y)[0, 1]
            except:
                print("preds", preds)
                print("y", y)
                raise

            if self.options["verbose"]:
                print(
                    f"\nPerformance of {self.typeof} model:\n"
                    f"\tRoot Mean Squared Error (RMSE): {rmse}\n"
                    f"\tMean Absolute Error (MAE): {mae}\n"
                    f"\tR^2 Score: {r2}"
                )

            # Reshape the data for sklearn
            all_labels = np.array(y).reshape(-1, len(self.target_var))
            all_preds = np.array(preds).reshape(-1, len(self.target_var))

            for target, target_var in enumerate(self.target_var):
                # Use a model to generate the line of best fit
                ridge = Ridge(alpha=0.1)
                ridge.fit(
                    all_labels[:, target].reshape(-1, 1),
                    all_preds[:, target].reshape(-1, 1),
                )
                x_new = np.linspace(
                    np.min(all_labels[:, target]), np.max(all_labels[:, target]), 100
                ).reshape(-1, 1)
                y_new = ridge.predict(x_new)

                # Plot the predictions vs the true values to see how well the model did
                plt.plot(all_labels[:, target], label="true")
                plt.plot(all_preds[:, target], label="prediction")
                plt.title(f"Observations vs Values of {self.name} for {target_var}")
                plt.xlabel("Observations")
                plt.ylabel("Values")
                plt.legend()
                plt.savefig(
                    "/".join(
                        [
                            self.output_dirs["performance"],
                            f"line_{target_var}.png",
                        ]
                    )
                )
                if self.options["verbose"]:
                    plt.show()

                plt.figure(figsize=(8, 8))
                plt.scatter(all_labels[:, target], all_preds[:, target], alpha=0.3)
                plt.title(f"True Values vs Predictions of {self.name} for {target_var}")
                plt.xlabel("True Values")
                plt.ylabel("Predictions")
                plt.axis("equal")
                plt.axis("square")
                maximum = max(plt.xlim()[1], plt.ylim()[1])
                minimum = min(plt.xlim()[0], plt.ylim()[0])
                lims = [minimum, maximum]
                plt.xlim(lims)
                plt.ylim(lims)

                # Line of best fit
                plt.plot(x_new, y_new, "r--")

                # Line of perfect fit
                plt.plot(lims, lims, "k-")
                plt.savefig(
                    "/".join(
                        [
                            self.output_dirs["performance"],
                            f"scatter_{target_var}.png",
                        ]
                    )
                )
                if self.options["verbose"]:
                    plt.show()
                    print(f"Correlation Coefficient: {corr}")

            self.r2 = float(r2)
            self.rmse = float(rmse)
            self.mae = float(mae)
            self.corr = float(corr)
