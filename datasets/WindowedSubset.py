from torch.utils.data.dataset import Subset
from torch._utils import _accumulate
from torch import randperm

import random

def windowed_random_split(dataset, lengths):
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths)).tolist()
    return [WindowedSubset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]

class WindowedSubset(Subset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.coin_indices = indices

        self.indices = self.dataset.full_len_for_indices(indices)
        random.shuffle(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
