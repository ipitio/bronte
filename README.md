# Bronte

From Greek βροντή (brontḗ, “thunder”).

![thunder](thunder.png)

This was started before realizing `PyTorch Lightning` exists... Good thing engineering is fun! The notebook `bronte.ipynb` features an ETL Pipeline for, and Deep Learning using, the Bronte framework, with an example dataset of Basketball statistics. This project is intended to be a tutorial and a playground and is definitely a work in progress!

## ETL Pipeline

First we extract the data, from CSVs in this case, merge them, and perform EDA using `ydata-profiling`. Then we transform the data with some standard cleaning and dataset-specific feature engineering. Finally, we partition the data into small chunks and load it into a database.

## Deep Learning

This database is then read table-by-table, for each task and arch specified, and passed to the `Trainer`. Over the course of training, checkpoints, metrics, and the model's state and options from the best epoch will be saved to `runs/models`.

### Framework

The framework is designed to be modular and extensible. It is based on `PyTorch` and contains the following modules:

- `data`: Datasets
- `loss`: Loss calculations
- `arch`: Model architectures (i.e. layers)
- `task`: Task-specific logic (e.g. preprocessing, evaluation)
- `core`: Base model class, training loop
- `main`: Model factory, starts training

The `Trainer` class in the `main` module takes a dictionary of parameters, including task and arch, and creates a model. When data is passed to the `Trainer`, it splits it into features X and target(s) y, and passes these to the model's `fit` method, which then initializes the layers, optimizer, scheduler, criterion, scaler, datasets, and dataloaders, and starts training.

#### Supports

- Hyperparameter tuning with `optuna` (incl. pruning)
- State and layer checkpointing
- Gradient accumulation, clipping, scaling
- Mixed precision training with `autocast`
- Distributed training with `dask` and `dask.distributed`
- Feature importance with `shap`
- Progress bars with `tqdm`
- Single/multi input/output

#### TODO

- [ ] Logging with `tensorboard`
- [ ] TPU support
- [ ] More everything
- [ ] Tests
- [ ] Documentation
- [ ] Resume from checkpoint

## Inference

An example is provided at the end of the notebook of the `Trainer` being used to load trained models, which can be used to make predictions on new data.
