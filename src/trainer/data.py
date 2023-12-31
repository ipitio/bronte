import torch
import dask.dataframe as dd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset


class DeepDataset(IterableDataset):
    def __init__(
        self,
        X,
        y,
        sequence_length,
        batch_size,
        device,
        num_workers,
        client=None,
        shuffle=False,
        drop_last=False,
    ):
        super(DeepDataset, self).__init__()
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        self.step = batch_size
        self.device = device
        self.shuffle = shuffle
        self.length = len(self.X)
        self.num_workers = max(num_workers, 1)
        self.drop_last = drop_last
        self.num_batches = self.__len__()
        self.num_processed = -1

        if client is not None and isinstance(self.X, dd.DataFrame):
            self.X = client.persist(X)
            self.y = client.persist(y)

    def to_tensor(self, X, y):
        if not isinstance(X, np.ndarray) and not isinstance(X, torch.Tensor):
            X = X.to_numpy()
            y = y.to_numpy()
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
            y = torch.from_numpy(y).float()
        return X.to(self.device), y.to(self.device).reshape(-1, len(self.y.columns))

    def __len__(self):
        if self.sequence_length > 0 and self.sequence_length < self.step:
            num_sequences = self.length - self.sequence_length + 1
            num_batches = num_sequences // self.step
            if not self.drop_last and num_sequences % self.step != 0:
                num_batches += 1
        else:
            num_batches = self.length // self.step
            if not self.drop_last and self.length % self.step != 0:
                num_batches += 1
        return num_batches

    def __iter__(self):
        buffer = []
        start = 0
        end = self.length
        X = self.X
        y = self.y

        if isinstance(self.X, dd.DataFrame):
            X = X.compute()
            y = y.compute()
        else:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = (
                0 if worker_info is None or self.num_workers < 2 else worker_info.id
            )
            index = self.length // self.num_workers + (
                self.length % self.num_workers > 0
            )
            start = worker_id * index
            end = min(start + index, self.length)

        indices = np.arange(start, end)
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(0, len(indices), self.step):
            batch_indices = indices[i : i + self.step].tolist()

            if self.sequence_length <= 0 or self.sequence_length >= self.step:
                X_batch = X.iloc[batch_indices]
                y_batch = y.iloc[batch_indices]
            else:
                # Add buffer items to the current batch
                batch_indices = buffer + batch_indices
                buffer = []

                sequences = []
                for j in range(0, len(batch_indices), self.sequence_length):
                    sequence = batch_indices[j : j + self.sequence_length]
                    if len(sequence) == self.sequence_length:
                        sequences.append(sequence)
                    else:
                        buffer = sequence

                # Gather data for all sequences and then compute
                X_batch = np.array([X.iloc[seq] for seq in sequences])
                y_batch = np.array([y.iloc[seq] for seq in sequences])

            self.num_processed += 1
            if len(X_batch) > 0:
                if self.num_processed < self.num_batches or (
                    not self.drop_last and buffer
                ):
                    yield self.to_tensor(X_batch, y_batch)
                else:
                    return self.to_tensor(X_batch, y_batch)

        # Handle leftover buffer at the end
        padded_X = pad_sequence([torch.tensor(X.iloc[buffer].to_numpy())], True)
        padded_y = pad_sequence([torch.tensor(y.iloc[buffer].to_numpy())], True)
        return self.to_tensor(padded_X, padded_y)
