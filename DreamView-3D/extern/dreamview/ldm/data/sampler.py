from torch.utils.data.distributed import DistributedSampler
import math

import torch
import torch.distributed as dist
import random


class Combined2DAnd3DSampler(DistributedSampler):
    def __init__(self, dataset, batch_size, num_replicas=None,
                 rank=None, shuffle=True,
                 seed=0, drop_last=False) -> None:
        # super.__init__()
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError("Invalid rank {}, rank should be in the interval"
                             " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas  # number of processes
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        self.len_dataset_2d = len(self.dataset) - self.dataset.split_length
        self.len_dataset_3d = self.dataset.split_length

        # processing 2d dataset
        if self.drop_last and self.len_dataset_2d % self.num_replicas != 0:
            self.num_samples_2d = math.ceil((self.len_dataset_2d - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples_2d = math.ceil(self.len_dataset_2d / self.num_replicas)
        self.total_size_2d = self.num_samples_2d * self.num_replicas

        # processing 3d dataset
        if self.drop_last and self.len_dataset_3d % self.num_replicas != 0:
            self.num_samples_3d = math.ceil((self.len_dataset_3d - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples_3d = math.ceil(self.len_dataset_3d / self.num_replicas)
        self.total_size_3d = self.num_samples_3d * self.num_replicas

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices_2d = (torch.randperm(self.len_dataset_2d, generator=g) + self.len_dataset_3d).tolist()
            indices_3d = (torch.randperm(self.len_dataset_3d, generator=g)).tolist()
        else:
            indices_2d = list(range(self.len_dataset_2d))
            indices_2d = [i + self.len_dataset_3d for i in indices_2d]
            indices_3d = list(range(self.len_dataset_3d))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size_2d - len(indices_2d)
            if padding_size <= len(indices_2d):
                indices_2d += indices_2d[:padding_size]
            else:
                indices_2d += (indices_2d * math.ceil(padding_size / len(indices_2d)))[:padding_size]

            padding_size = self.total_size_3d - len(indices_3d)
            if padding_size <= len(indices_3d):
                indices_3d += indices_3d[:padding_size]
            else:
                indices_3d += (indices_3d * math.ceil(padding_size / len(indices_3d)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices_2d = indices_2d[:self.total_size_2d]
            indices_3d = indices_3d[:self.total_size_3d]
        assert len(indices_2d) == self.total_size_2d and len(indices_3d) == self.total_size_3d

        # subsample
        indices_2d = indices_2d[self.rank:self.total_size_2d:self.num_replicas]
        indices_3d = indices_3d[self.rank:self.total_size_3d:self.num_replicas]
        assert len(indices_2d) == self.num_samples_2d and len(indices_3d) == self.num_samples_3d

        # mixing the 2d and 3d data
        final_indices = []
        it_3d = it_2d = 0
        while it_3d < len(indices_3d) and it_2d < len(indices_2d):
            if random.random() > 0.3:
                final_indices += indices_3d[it_3d:it_3d + self.batch_size]
                it_3d += self.batch_size
            else:
                final_indices += indices_2d[it_2d:it_2d + self.batch_size]
                it_2d += self.batch_size

        self.num_samples = len(final_indices)
        return iter(final_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
