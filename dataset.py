import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt
import pandas 
import h5py

class CoinDataset(Dataset):
	def __init__(self, path_to_hdf5, examples, max_length):
		self.path_to_hdf5 = path_to_hdf5
		self.examples = examples
		self.max_length = max_length

		self.data_file = None
		self.cuda_available = torch.cuda.is_available()

		self.data_cache = []

		for i in range(len(examples)):
			print("Preparing {}/{}".format(i + 1, len(examples)), end="\r")
			self.data_cache.append(self.prepare_data_item(i))
		
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

	def prepare_data_item(self, idx):
		coin, gain, number = self.examples[idx]

		if not self.data_file:
			self.data_file = h5py.File(self.path_to_hdf5, "r")
		timeseries = self.data_file[coin][gain][number]["values"][:][:self.max_length:8] # Diese Art der Indizierung ist schneller

		# self.data_file.close()

		timeseries = self.min_max_scale(timeseries)
		reversed_timeseries = np.flip(timeseries).copy() # Negative strides nicht supported von pytorch, deshalb copy()
		teacher_input = np.zeros(reversed_timeseries.shape)
		teacher_input[1:] = reversed_timeseries[1:]
		teacher_input[0] = -1

		reversed_timeseries = self.convert_to_tensor(reversed_timeseries).view(reversed_timeseries.size, 1)
		teacher_input = self.convert_to_tensor(teacher_input).view(teacher_input.size, 1)
		timeseries = self.convert_to_tensor(timeseries).view(timeseries.size, 1)
		coin_class = self.convert_to_tensor(np.array(self.get_class_for_coin(int(coin))))

		return {"input": timeseries, "reversed_input": reversed_timeseries, "teacher_input": teacher_input ,"label": coin_class}

	def __getitem__(self, idx):
		return self.data_cache[idx]
		# coin, gain, number = self.examples[idx]

		# if not self.data_file:
		# 	self.data_file = h5py.File(self.path_to_hdf5, "r")
		# timeseries = self.data_file[coin][gain][number]["values"][:][:self.max_length:8] # Diese Art der Indizierung ist schneller

		# # self.data_file.close()

		# timeseries = self.min_max_scale(timeseries)
		# reversed_timeseries = np.flip(timeseries).copy() # Negative strides nicht supported bon pytorch, deshalb copy
		# teacher_input = np.zeros(reversed_timeseries.shape)
		# teacher_input[1:] = reversed_timeseries[1:]

		# reversed_timeseries = self.convert_to_tensor(reversed_timeseries).view(reversed_timeseries.size, 1)
		# teacher_input = self.convert_to_tensor(teacher_input).view(teacher_input.size, 1)
		# timeseries = self.convert_to_tensor(timeseries).view(timeseries.size, 1)
		# coin_class = self.convert_to_tensor(np.array(self.get_class_for_coin(int(coin))))

		# return {"input": timeseries, "reversed_input": reversed_timeseries, "teacher_input": teacher_input ,"label": coin_class}

class CoinDatasetLoader:
	def __init__(self, path_to_hdf5, validation_split, test_split):
		self.path_to_hdf5 = path_to_hdf5

		self.data_file = h5py.File(path_to_hdf5, "r")

		self.training_paths = []
		self.validation_paths = []
		self.test_paths = []

		self.shortest_seq = np.inf

		# Load states for training, validation and test set
		for coin in self.data_file.keys():
			for gain in self.data_file[coin].keys():
				example_set = np.array(list(self.data_file[coin][gain]))

				self.shortest_seq = np.minimum(self.shortest_seq_in_example_set(coin, gain, example_set), self.shortest_seq) 

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

		self.shortest_seq = int(self.shortest_seq)

		print("Shortest sequence length: {}".format(self.shortest_seq))

	def shortest_seq_in_example_set(self, coin, gain, example_set):
		shortest = np.inf

		for num in example_set:
			seq_len = self.data_file[coin][gain][num]["values"].size
			shortest = np.minimum(shortest, seq_len)

		return shortest

	def get_num_training_examples(self):
		return len(self.training_examples)

	def get_num_validation_examples(self):
		return len(self.validation_examples)

	def get_num_test_examples(self):
		return len(self.test_examples)

	def get_dataset(self, mode):
		if mode == "training":
			return CoinDataset(path_to_hdf5=self.path_to_hdf5, examples=self.training_paths, max_length=self.shortest_seq)
		elif mode == "validation":
			return CoinDataset(path_to_hdf5=self.path_to_hdf5, examples=self.validation_paths, max_length=self.shortest_seq)
		elif mode == "test":
			return CoinDataset(path_to_hdf5=self.path_to_hdf5, examples=self.test_paths, max_length=self.shortest_seq)

		raise Exception("Unknown mode: {}".format(mode))