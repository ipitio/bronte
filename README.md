# Bronte

From Greek βροντή (brontḗ, “thunder”).

![thunder](thunder.png)

The notebook `bronte.ipynb` features an ETL Pipeline for, and Deep Learning using, the modular and extensible `Bronte` framework, with an example dataset of Basketball statistics. This project is intended to be a tutorial and a playground and is definitely a work in progress!

## ETL Pipeline

First we extract the data, from CSVs in this case, merge them, and perform EDA using `ydata-profiling`. Then we transform the data with some standard cleaning and dataset-specific feature engineering. Finally, we partition the data into small chunks and load it into a database.

## Deep Learning

This database is then read table-by-table, for each task and arch specified, and passed to `Bronte`. Over the course of training, checkpoints, metrics, and the model's state and options from the best epoch will be saved to `models/`.

### Framework

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

    trainer = Bronte(task | arch)
    trainer.fit(X, y)
    y_new = trainer.predict(X_new)

> #### Note
>
> You must initialize the layers not in `__init__`, but in `init_layers`, as this is used to (re)initialize the model's layers when (resuming) training.

#### Supports

- CPU, GPU
- Hyperparameter tuning with `optuna`
  - Persistent
  - Pruning
- Gradient:
  - Accumulation
  - Clipping
  - Scaling (GPU-only)
- Training:
  - Persistent
  - Mixed Precision
  - Single- and multi-input (multi-task) and -output (multi-class)
  - Learning Rate scheduling
  - Transfer Learning:
    - Freeze layers
    - Parameter transfer
    - Fine-tuning
  - Parallel and Distributed with `dask`
- Arch:
  - FFN, RNN with Attention
  - Checkpointing (GPU-only)
- Task:
  - Regression
  - Classification
- Calculating feature importances with `shap`

#### TODO

- [ ] Logging with `tensorboard`
- [ ] TPU support
- [ ] Frontend + Flask
- [ ] More archs, tasks
- [ ] Tests
- [ ] Documentation

## Inference

An example is provided at the end of the notebook of `Bronte` being used to load trained models, which can be used to make predictions on new data.

    trainer = Bronte(path="path/to/model.pt")
    y_new = trainer.predict(X_new)
