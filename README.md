# Bronte

![thunder](thunder.png)

[![License: AGPL](https://img.shields.io/badge/License-AGPL-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

## Framework

`Bronte` views a model not as layers, but a trainer of layers, whose preprocessing and evaluation are task-specific. Like with `Pytorch Lightning`, this abstracts away training and allows for a clean separation of concerns, making it easy to modify, add, and experiment with different tasks and architectures. If you'd like to add a new task or architecture, you can do so by creating a new class in the appropriate module and adding it in `main`.

It is composed of the following modules:

- `main`: Factory
- `arch`: Architecture
- `task`: Preprocessing and Evaluation
- Trainer:
  - `base`: Training
  - `data`: Datasets
  - `loss`: Loss calculations
  - `tune`: Hyperparameter tuner

`Bronte` takes a dictionary of options, including the names of a task and an arch, and creates a model. When data is passed to `Bronte`, it splits it into features X and target(s) y, and passes these to the model's `fit` method, which then initializes the layers, optimizer, scheduler, criterion, scaler, datasets, and dataloaders, and starts training. Please look at the notebook for a list of all the options (under Deep Learning > Options).

> ### Note
>
> You must initialize the layers not in `__init__`, but in `init_layers`, as this is used to (re)initialize the model's layers when (resuming) training.

## Usage

### Training

    trainer = Bronte(task | arch)
    trainer.fit(data)

### Inference

    trainer = Bronte(path="model.pt")
    y = trainer.predict(X)

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

The notebook `basketball.ipynb` features an ETL Pipeline for, and Deep Learning using, the modular and extensible `Bronte` framework, with an example dataset of Basketball statistics.

### ETL Pipeline

First, we extract the data, from CSVs in this case, merge them, and perform EDA using `ydata-profiling`. Then we transform the data with some standard cleaning and dataset-specific feature engineering. Finally, we partition the data into small chunks and load it into a database.

### Training

This database is then read table-by-table, for each task and arch specified, and passed to `Bronte`. Over the course of training, checkpoints of state and visuals of metrics and importances will be saved to `models/`.

### Inference

An example is provided at the end of the notebook of `Bronte` being used to load trained models, which can make predictions on new data.
