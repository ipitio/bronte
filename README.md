# Bronte

From Greek βροντή (brontḗ, “thunder”).

![thunder](thunder.png)

This was started before realizing `PyTorch Lightning` exists... Good thing engineering is fun! The notebook `bronte.ipynb` features an ETL Pipeline for, and Deep Learning using, the Bronte framework, with an example dataset of Basketball statistics. This project is intended to be a tutorial and a playground and is definitely a work in progress!

## ETL Pipeline

First we extract the data, from CSVs in this case, merge them, and perform EDA using `ydata-profiling`. Then we transform the data with some standard cleaning and dataset-specific feature engineering. Finally, we partition the data into small chunks and load it into a database.

## Deep Learning

This database is then read table-by-table, for each task and arch specified, and passed to `Bronte`. Over the course of training, checkpoints, metrics, and the model's state and options from the best epoch will be saved to `models/`.

### Framework

The framework is designed to be modular and extensible. It is based on `PyTorch` and contains the following modules:

- `data`: Datasets
- `loss`: Loss calculations
- `tune`: Tuner class
- `core`: Model components: base, task, arch
- `main`: Model factory

The `Bronte` takes a dictionary of parameters, including task and arch, and creates a model. When data is passed to `Bronte`, it splits it into features X and target(s) y, and passes these to the model's `fit` method, which then initializes the layers, optimizer, scheduler, criterion, scaler, datasets, and dataloaders, and starts training.

If you'd like to add a new task or arch, you can do so by creating a new class in the appropriate module.

> #### Note
>
> You must initialize the layers not in `__init__`, but in `init_layers`, as this is used to (re)initialize the model's layers when (resuming) training.

#### Supports

- CPU, GPU
- Single and multiple inputs and outputs
- Learning Rate scheduling
- Hyperparameter tuning with `optuna`
  - Persistent
  - Pruning
- Gradient:
  - Accumulation
  - Clipping
  - Scaling
- Training:
  - Mixed Precision
  - Persistent/Transfer Learning
  - Parallel and Distributed with `dask`
- Calculating feature importances with `shap`

#### TODO

- [ ] Logging with `tensorboard`
- [ ] TPU support
- [ ] Frontend + Flask
- [ ] More archs
- [ ] Tests
- [ ] Documentation

## Inference

An example is provided at the end of the notebook of `Bronte` being used to load trained models, which can be used to make predictions on new data.
