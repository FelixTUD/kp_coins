import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt
import pandas 
import h5py



class CoinDataset(Dataset):
	def __init__(self, path_to_hdf5, examples):
		self.path_to_hdf5 = path_to_hdf5
		self.examples = examples

		self.data_file = h5py.File(path_to_hdf5, "r")
		self.cuda_available = torch.cuda.is_available()
		
	def __len__(self):
		return len(self.examples)

	def get_class_for_coin(self, coin):
		if coin == 1:
			return 0
		elif coin == 2:
			return 1
		elif coin == 5:
			return 2
		elif coin == 20:
			return 3
		elif coin == 50:
			return 4
		elif coin == 100:
			return 5
		elif coin == 200:
			return 6

		raise Exception("Unknown coin type: {}".format(coin))

	def standard_scale(self, timeseries):
		mean = timeseries.mean()
		std = timeseries.std()
		return (timeseries - mean) / std

	def min_max_scale(self, timeseries):
		min = np.min(timeseries)
		max = np.max(timeseries)
		return (timeseries - min) / (max - min)

	def convert_to_tensor(self, data):
		tensor = torch.from_numpy(data).float()

		if self.cuda_available:
			tensor = tensor.cuda()

		return tensor

	def __getitem__(self, idx):
		coin, gain, number = self.examples[idx]

		timeseries = self.data_file[coin][gain][number]["values"][:]
		timeseries = self.min_max_scale(timeseries)

		timeseries = self.convert_to_tensor(timeseries).view(timeseries.size, 1)
		coin_class = self.convert_to_tensor(np.array(self.get_class_for_coin(int(coin))))

		return {"input": timeseries, "label": coin_class}

class CoinDatasetLoader:
	def __init__(self, path_to_hdf5, validation_split, test_split):
		self.path_to_hdf5 = path_to_hdf5

		self.data_file = h5py.File(path_to_hdf5, "r")

		self.training_paths = []
		self.validation_paths = []
		self.test_paths = []

		# Load states for training, validation and test set
		for coin in self.data_file.keys():
			for gain in self.data_file[coin].keys():
				example_set = np.array(list(self.data_file[coin][gain]))

				num_validation_examples = int(len(example_set) * validation_split)
				num_test_examples = int(len(example_set) * test_split)
				num_training_examples = len(example_set) - num_validation_examples - num_test_examples

				validation_examples = np.random.choice(example_set, num_validation_examples, replace=False)
				example_set = np.delete(example_set, validation_examples)

				test_examples = np.random.choice(example_set, num_test_examples, replace=False)
				example_set = np.delete(example_set, test_examples)

				training_examples = example_set

				self.training_paths += [(coin, gain, x) for x in list(training_examples)]
				self.validation_paths += [(coin, gain, x) for x in list(validation_examples)]
				self.test_paths += [(coin, gain, x) for x in list(test_examples)]

	def get_num_training_examples(self):
		return len(self.training_examples)

	def get_num_validation_examples(self):
		return len(self.validation_examples)

	def get_num_test_examples(self):
		return len(self.test_examples)

	def get_dataset(self, mode):
		if mode == "training":
			return CoinDataset(path_to_hdf5=self.path_to_hdf5, examples=self.training_paths)
		elif mode == "validation":
			return CoinDataset(path_to_hdf5=self.path_to_hdf5, examples=self.validation_paths)
		elif mode == "test":
			return CoinDataset(path_to_hdf5=self.path_to_hdf5, examples=self.test_paths)

		raise Exception("Unknown mode: {}".format(mode))