import jax.numpy as jnp
import jax
import numpy as np
import pennylane as qml

class JAXStateDataset:
    def __init__(self, ideal_data):
        self.ideal_data = ideal_data

    def __getitem__(self, index):
        return self.ideal_data[index], 0

    def __len__(self):
        return len(self.ideal_data)

class JAXDataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True, drop_last=False, seed=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        if seed is not None:
            self.key = jax.random.PRNGKey(seed)
        else:
            self.key = jax.random.PRNGKey(0)
        
        self.indices = jnp.arange(len(self.dataset))
        self.reset()

    def reset(self):

        if self.shuffle:
            self.indices = jax.random.permutation(self.key, self.indices)
        self.current_idx = 0

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.current_idx >= len(self.indices):
            raise StopIteration

        end_idx = self.current_idx + self.batch_size
        batch_indices = self.indices[self.current_idx:end_idx]

        if self.drop_last and len(batch_indices) < self.batch_size:
            raise StopIteration

        samples = [self.dataset[i] for i in batch_indices]
        grouped = list(zip(*samples))
        batched = tuple(jnp.stack(g) for g in grouped)

        self.current_idx = end_idx
        return batched  # returns a tuple: (x_batch, y_batch, ...)
    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size