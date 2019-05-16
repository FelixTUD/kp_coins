import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt
import pandas 
import h5py
import random
from collections import defaultdict

class CoinDataset(Dataset):
	def __init__(self, path_to_hdf5, examples, max_length, shrink, args):
		self.path_to_hdf5 = path_to_hdf5
		self.examples = examples
		self.max_length = max_length
		self.shrink = shrink
		self.args = args
		self.use_rosa = args.rosa
		self.top_db = args.top_db

		self.data_file = None
		self.cuda_available = torch.cuda.is_available()

		self.data_cache = []
		self.preprocessed_data = []
		self.label_mapping = defaultdict(list)

		self.longest_timeseries = 0

		for i in range(len(examples)):
			print("#1: Preparing {}/{}".format(i + 1, len(examples)), end="\r")
			self.preprocess_data_step_1(i)

		if self.use_rosa:
			list_of_sizes = list(map(lambda x: (x[0], x[1]["input"].size), enumerate(self.preprocessed_data)))
			list_of_sizes = sorted(list_of_sizes, key=lambda x: x[1], reverse=True)

			self.preprocessed_data = list(map(lambda x: self.preprocessed_data[x[0]], list_of_sizes[5:-5]))

			self.longest_timeseries = list_of_sizes[5][1]

			print("Longest timeseries: {}".format(self.longest_timeseries))
		
		for i in range(len(self.preprocessed_data)):
			print("#2: Preparing {}/{}".format(i + 1, len(examples)), end="\r")
			self.preprocess_data_step_2(i)

		del self.preprocessed_data
		
	def __len__(self):
		return len(self.data_cache)

	def get_class_for_coin(self, coin):
		if self.args.debug:
			if coin == 5:
				return 0 
			elif coin == 100:
				return 1 
			
			raise Exception("Unknown coin type encountered in debug mode")
		else:
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

	def minus_one_plus_one_scale(self, timeseries):
		min = np.min(timeseries)
		max = np.max(timeseries)
		return (2*(timeseries - min) / (max - min)) - 1

	def convert_to_tensor(self, data):
		tensor = torch.from_numpy(data).float()

		if self.cuda_available:
			tensor = tensor.cuda()

		return tensor

	def preprocess_data_step_1(self, idx):
		coin, gain, number = self.examples[idx]

		if not self.data_file:
			self.data_file = h5py.File(self.path_to_hdf5, "r")

		timeseries = None
		if self.use_rosa:
			import librosa as rosa
			timeseries = self.data_file[coin][gain][number]["values"][:]
			timeseries = rosa.effects.trim(timeseries, top_db=self.top_db)[0][::self.shrink]
			# timeseries = rosa.util.normalize(timeseries)
			# timeseries = self.minus_one_plus_one_scale(timeseries)
		else:
			timeseries = self.data_file[coin][gain][number]["values"][:][:self.max_length:self.shrink] # Diese Art der Indizierung ist schneller
			timeseries = self.min_max_scale(timeseries)

		coin_class = np.array(self.get_class_for_coin(int(coin)))

		self.label_mapping[int(coin)].append(idx)

		self.preprocessed_data.append({"input": timeseries,"label": coin_class})

	def preprocess_data_step_2(self, idx):
		preprocessed = self.preprocessed_data[idx]
		timeseries, coin_class = preprocessed["input"], preprocessed["label"]

		if self.use_rosa:
			import librosa as rosa
			if self.args.batch_size > 1:
				timeseries = rosa.util.fix_length(timeseries, self.longest_timeseries)
			timeseries = self.minus_one_plus_one_scale(timeseries)

		reversed_timeseries = np.flip(timeseries).copy() # Negative strides (noch) nicht supported von pytorch, deshalb copy()
		teacher_input = np.zeros(reversed_timeseries.shape)
		teacher_input[1:] = reversed_timeseries[1:]
		teacher_input[0] = -1

		reversed_timeseries = self.convert_to_tensor(reversed_timeseries).view(reversed_timeseries.size, 1)
		teacher_input = self.convert_to_tensor(teacher_input).view(teacher_input.size, 1)
		timeseries = self.convert_to_tensor(timeseries).view(timeseries.size, 1)
		coin_class = self.convert_to_tensor(coin_class).long()

		self.data_cache.append({"input": timeseries, "reversed_input": reversed_timeseries, "teacher_input": teacher_input ,"label": coin_class})

	def old_prepare_data_item(self, idx):
		coin, gain, number = self.examples[idx]

		if not self.data_file:
			self.data_file = h5py.File(self.path_to_hdf5, "r")
		timeseries = self.data_file[coin][gain][number]["values"][:][:self.max_length:self.shrink] # Diese Art der Indizierung ist schneller

		if self.use_rosa:
			timeseries = rosa.effects.trim(timeseries, top_db=self.top_db)[0]
			timeseries = rosa.util.normalize(timeseries)
			timeseries = rosa.util.fix_length(timeseries, self.max_length//self.shrink)
		else:
			timeseries = self.min_max_scale(timeseries)
		reversed_timeseries = np.flip(timeseries).copy() # Negative strides (noch) nicht supported von pytorch, deshalb copy()
		teacher_input = np.zeros(reversed_timeseries.shape)
		teacher_input[1:] = reversed_timeseries[1:]
		teacher_input[0] = -1

		reversed_timeseries = self.convert_to_tensor(reversed_timeseries).view(reversed_timeseries.size, 1)
		teacher_input = self.convert_to_tensor(teacher_input).view(teacher_input.size, 1)
		timeseries = self.convert_to_tensor(timeseries).view(timeseries.size, 1)
		coin_class = self.convert_to_tensor(np.array(self.get_class_for_coin(int(coin)))).long()

		self.label_mapping[int(coin)].append(idx)

		return {"input": timeseries, "reversed_input": reversed_timeseries, "teacher_input": teacher_input ,"label": coin_class}

	def __getitem__(self, idx):
		return self.data_cache[idx]

	def get_data_for_coin_type(self, coin, num_examples):
		data = self.label_mapping[coin]
		data = list(map(lambda x: self[x], data[:num_examples]))

		return data
		
class CoinDatasetLoader:
	def __init__(self, path_to_hdf5, shrink, validation_split, test_split, args):
		# Debug mode only returns data for coins of values 5 and 100. Each with 100 training examples

		self.path_to_hdf5 = path_to_hdf5
		self.shrink = shrink
		self.args = args

		debug_mode = args.debug

		self.data_file = h5py.File(path_to_hdf5, "r")

		self.training_paths = []
		self.validation_paths = []
		self.test_paths = []

		self.shortest_seq =  700000

		# Load states for training, validation and test set
		coin_keys = self.data_file.keys() if not debug_mode else ["5", "100"]
		for coin in coin_keys:
			gain_keys = self.data_file[coin].keys()
			data_paths = []
			for gain in gain_keys:
				data_paths.extend([(gain, x) for x in self.data_file[coin][gain].keys()])

			num_validation_examples = int(len(data_paths) * validation_split) if not debug_mode else 20
			num_test_examples = int(len(data_paths) * test_split) if not debug_mode else 20
			num_training_examples = len(data_paths) - num_validation_examples - num_test_examples if not debug_mode else 120

			random.shuffle(data_paths)

			validation_examples = data_paths[:num_validation_examples]
			test_examples = data_paths[num_validation_examples:num_validation_examples+num_test_examples]
			training_examples = data_paths[num_validation_examples+num_test_examples:num_validation_examples+num_test_examples+num_training_examples]

			self.training_paths += [(coin,) + x for x in training_examples]
			self.validation_paths += [(coin,) + x for x in validation_examples]
			self.test_paths += [(coin,) + x for x in test_examples]

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
			return CoinDataset(path_to_hdf5=self.path_to_hdf5, examples=self.training_paths, max_length=self.shortest_seq, shrink=self.shrink, args=self.args)
		elif mode == "validation":
			return CoinDataset(path_to_hdf5=self.path_to_hdf5, examples=self.validation_paths, max_length=self.shortest_seq, shrink=self.shrink, args=self.args)
		elif mode == "test":
			return CoinDataset(path_to_hdf5=self.path_to_hdf5, examples=self.test_paths, max_length=self.shortest_seq, shrink=self.shrink, args=self.args)

		raise Exception("Unknown mode: {}".format(mode))
