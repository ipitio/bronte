# Bronte

![thunder](thunder.png)

[![License: AGPL](https://img.shields.io/badge/License-AGPL-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

`bronte` is a modular and extensible Deep Learning framework; It views a model not as layers, but a trainer of layers, whose preprocessing and evaluation are task-specific. Like with `Pytorch Lightning`, this abstracts away training and allows for a clean separation of concerns, making it easy to modify, add, and experiment with different tasks and architectures. If you'd like to add a new one, you can do so by creating a new class in the appropriate module and adding it in `bronte`.

It is composed of the following modules:

- `bronte`: Factory and Driver
- `arch`: Layers and Forward Pass
- `task`: Preprocessing and Evaluation
- `base`: Training and Inference
- `data`: Datasets
- `loss`: Loss calculations
- `tune`: Hyperparameter tuning

`Bronte` the class takes a dictionary of options, including the names of a task and an arch, and creates a model. When data is passed to `Bronte`, it splits it into features X and target(s) y, and passes these to the model's `fit` method, which then initializes the layers, optimizer, scheduler, criterion, scaler, datasets, and dataloaders, and starts training. Please look at the notebook for a list of all currently supported options (under Deep Learning > Options).

> **Note**
>
> You must initialize the layers not in `__init__`, but in `init_layers`, as this is used to (re)initialize the model's layers when (resuming) training.

## Usage

### Training

    import bronte

    data = [df]
    models = [task | arch]

    # load data into tables
    for df in data:
      bronte.load(df)

    # start tensorboard
    bronte.track()

    # train models on tables, returning list of Bronte objects
    trainers = bronte.fit(models)

    # call again to stop tensorboard
    bronte.track()

    # flush db
    bronte.flush()

### Inference

    import bronte

    XX = [X]
    paths = ["models/.../model.pt"]

    # predict on list of new data, returning dict: {path: {str(XX.index(X)): y}}
    predictions = bronte.predict(XX, paths)

## Supports

- Training:
  - (C/G/T)PU
  - Persistence
  - Mixed Precision
  - Multi input and output
  - Model and state checkpointing
  - Learning Rate scheduling
  - Transfer Learning
  - Gradient accumulation and scaling
  - Parallel and Distributed with `dask`
  - Hyperparameter tuning with `optuna`
  - Calculating feature importances with `shap`
  - Monitoring/Logging with `tensorboard`
- Tasks:
  - Regression
  - Classification
- Architectures:
  - FFN
  - RNN with Attention

## TODO

- [ ] Frontend + Flask
- [ ] More archs, tasks
- [ ] Tests, Typing, Documentation

## Example

The notebook `basketball.ipynb` runs an ETL Pipeline for a sample dataset of Basketball statistics and performs Deep Learning using `Bronte`.

### ETL Pipeline

First, the data is extracted (from CSVs in this case), merged, and examined (ie. EDA) with `ydata-profiling`. Then it's transformed with some standard cleaning and dataset-specific feature engineering, before being partitioned into small chunks and loaded into a database.

### Deep Learning

This database is then read table-by-table, for each task and arch specified, and passed to `Bronte`. Over the course of training, checkpoints of state and visuals of metrics and importances will be saved to `models/`. Once training is complete, `Bronte` can be used to load the trained models and make predictions on new data.
