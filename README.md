# Bronte

![thunder](thunder.png)

[![License: AGPL](https://img.shields.io/badge/License-AGPL-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

`Bronte` is a modular and extensible Deep Learning framework; It views a model not as layers, but as a trainer of layers, whose preprocessing and evaluation are task-specific. Like with `Pytorch Lightning`, this abstracts away training and allows for a clean separation of concerns, making it easy to modify, add, and experiment with different tasks and architectures. If you'd like to add a new task or architecture, you can do so by creating a new class in the appropriate module and adding it in `main`.

It is composed of the following modules:

- `bronte`: Factory
- `arch`: Architecture
- `task`: Preprocessing and Evaluation
- Trainer:
  - `base`: Training
  - `data`: Datasets
  - `loss`: Loss calculations
  - `tune`: Hyperparameter tuner

`Bronte` takes a dictionary of options, including the names of a task and an arch, and creates a model. When data is passed to `Bronte`, it splits it into features X and target(s) y, and passes these to the model's `fit` method, which then initializes the layers, optimizer, scheduler, criterion, scaler, datasets, and dataloaders, and starts training. Please look at the notebook for a list of all the options (under Deep Learning > Options).

> **Note**
>
> You must initialize the layers not in `__init__`, but in `init_layers`, as this is used to (re)initialize the model's layers when (resuming) training.

## Usage

### Training

#### Single

    from bronte import Bronte
    model = task | arch
    trainer = Bronte(model)
    trainer.fit(data)

#### Batch

    import bronte
    bronte.flush() # flush db
    bronte.load(data) # load data into a table
    models = [task | arch, task2 | arch2]
    trainers = bronte.fit(models) # train models on all tables

### Inference

    trainer = Bronte(path="models/.../model.pt")
    y = trainer.predict(X)
    # or
    trainers = ["models/.../model.pt", "models/.../model2.pt"]
    ys = bronte.predict(X, trainers)

## Features

- Training:
  - (C/G/T)PU
  - Persistent
  - Mixed Precision
  - Multi-task
  - Model and state checkpointing
  - Learning Rate scheduling
  - Transfer Learning
  - Gradient accumulation and scaling
  - Parallel and Distributed with `dask`
  - Hyperparameter tuning with `optuna`
  - Calculating feature importances with `shap`
  - Logging with `tensorboard`
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

The notebook `basketball.ipynb` runs an ETL Pipeline for, and performs Deep Learning using, `Bronte`, with a sample dataset of Basketball statistics.

### ETL Pipeline

First, the data is extracted (from CSVs in this case), merged, and examined (ie. EDA) with `ydata-profiling`. Then it's transformed with some standard cleaning and dataset-specific feature engineering, before being partitioned into small chunks and loaded into a database.

### Deep Learning

This database is then read table-by-table, for each task and arch specified, and passed to `Bronte`. Over the course of training, checkpoints of state and visuals of metrics and importances will be saved to `models/`. Once training is complete, `Bronte` can be used to load the trained models and make predictions on new data.
