import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

import numpy as np
import pandas 
import h5py
import random
from collections import defaultdict

class WindowedDataset(Dataset):
	def __init__(self, args):
		import librosa as rosa 
		self.rosa = rosa 		# Multiprocessing bullshit

		print("Loading the dataset with windows.")

		self.shrink = args.shrink
		assert(self.shrink > 0)

		self.architecture = args.architecture
		self.top_db = args.top_db
		self.path_to_hdf5 = args.path
		self.use_cuda = torch.cuda.is_available()
		self.preloaded_data = defaultdict(list)
		self.coin_mapping = defaultdict(list)

		self.window_size = args.window_size
		assert(self.window_size > 0)
		
		self.window_gap = args.window_gap
		assert(self.window_gap > 0)
		self.total_windows = 0

		self.data_file = h5py.File(self.path_to_hdf5, "r")

		if args.coins:
			assert(len(args.coins) > 0)
			self.coins = list(map(str, args.coins))
		else:
			self.coins = list(self.data_file.keys())
		print("Sampling from coin classes: {}".format(self.coins)) 

		if args.num_examples:
			assert (args.num_examples <= self.get_min_number_of_coins())
			self.min_num_coins = args.num_examples
		else:
			self.min_num_coins = self.get_min_number_of_coins()

		print("Sampling equal number of coins for each class: {} examples".format(self.min_num_coins))
		
		self.sample_space = self.get_sample_space()

		self.max_length = 0
		for coin, values in self.sample_space.items():
			for gain, example in values:
				self.max_length = max(len(self.data_file[coin][gain][example]["values"]) // self.shrink, self.max_length)
		print("Maximum timeseries length: {}".format(self.max_length))

		print("Preloading data to {} memory ...".format("gpu" if self.use_cuda else "cpu"))
		self.preload_samples()
		print("Loaded {} windows".format(self.total_windows))

		print("Shuffeling preloaded data ...")
		for key, value in self.preloaded_data.items():
			random.shuffle(value)
			self.preloaded_data[key] = value

		self.generate_coin_mapping_index()

	def generate_coin_mapping_index(self):
		self.coin_mapping = defaultdict(list)
		for _, samples in self.preloaded_data.items():
			for coin, data in samples:
				self.coin_mapping[coin].append(data)
		self.min_mapped_samples = min([len(samples) for _, samples in self.coin_mapping.items()])
		print("Minimum Number of window samples per coin: {}".format(self.min_mapped_samples))

	def get_num_coins_per_class(self):
		return self.min_mapped_samples

	def get_num_loaded_coins(self):
		return len(self.coins)

	def get_max_length(self):
		return self.max_length

	def preprocess_time_series(self, timeseries):
		timeseries = self.rosa.effects.trim(timeseries, top_db=self.top_db)[0][::self.shrink]

		if timeseries.size < self.window_size:
			timeseries = self.rosa.util.fix_length(timeseries, self.window_size)

		min = np.min(timeseries)
		max = np.max(timeseries)
		return (2*(timeseries - min) / (max - min)) - 1

	def convert_to_tensor(self, data):
		tensor = torch.from_numpy(data).float()

		if self.use_cuda:
			tensor = tensor.cuda()

		return tensor

	def convert_to_one_hot_index(self, coin):
		return np.array(self.coins.index(coin))

	def generate_data(self, timeseries, coin):
		reversed_timeseries = np.flip(timeseries).copy() # Negative strides (noch) nicht supported von pytorch, deshalb copy()
		teacher_input = np.zeros(reversed_timeseries.shape)
		teacher_input[1:] = reversed_timeseries[1:]
		teacher_input[0] = -1

		reversed_timeseries = self.convert_to_tensor(reversed_timeseries)
		teacher_input = self.convert_to_tensor(teacher_input)
		timeseries = self.convert_to_tensor(timeseries)

		if self.architecture in ["enc_dec", "simple_rnn"]:
			reversed_timeseries = reversed_timeseries.unsqueeze(1)
			teacher_input = teacher_input.unsqueeze(1)
			timeseries = timeseries.unsqueeze(1)

		coin_class = self.convert_to_tensor(self.convert_to_one_hot_index(coin)).long()
		return {"input": timeseries, "reversed_input": reversed_timeseries, "teacher_input": teacher_input ,"label": coin_class}

	def preload_samples(self):
		global_index = 0
		for coin, samples in self.sample_space.items():
			for index, (gain, example) in enumerate(samples):
				print("\rPreloading coin {}: {}/{}".format(coin, index + 1, len(samples)), end="")

				timeseries = self.data_file[coin][gain][example]["values"][:]
				timeseries = self.preprocess_time_series(timeseries)

				if timeseries.size == self.window_size:
					self.preloaded_data[global_index].append((coin, self.generate_data(timeseries, coin)))
					self.total_windows += 1
					global_index += 1
				else:
					for i in range(0, timeseries.size - self.window_size, self.window_gap):
						window = timeseries[i:i+self.window_size]
						self.preloaded_data[global_index].append((coin, self.generate_data(window, coin)))
						self.total_windows += 1
					global_index += 1

			print("")

	def get_sample_space(self):
		coin_samples = defaultdict(list)

		for coin in self.coins:
			current_samples = []

			for gain in self.data_file[coin].keys():
				for example in self.data_file[coin][gain]:
					current_samples.append((gain, example))

			coin_samples[coin] = random.sample(current_samples, self.min_num_coins)

		return coin_samples

	def get_min_number_of_coins(self):
		num_coin_examples = defaultdict(lambda: 0)

		for coin in self.coins:
			for gain in self.data_file[coin].keys():
				num_coin_examples[coin] += len(self.data_file[coin][gain].keys())

		min_num_coins_class = min(num_coin_examples, key=num_coin_examples.get)
		min_num_coins = num_coin_examples[min_num_coins_class]
		del num_coin_examples

		return min_num_coins

	def __len__(self):
		return len(self.preloaded_data)
		
	def __getitem__(self, idx):
		return random.choice(self.preloaded_data[idx])[1]

	def get_data_for_coin_type(self, coin, num_examples):
		result = []
		if type(coin) != str:
			coin = str(coin)

		samples = self.coin_mapping[coin]
		
		assert(len(samples) >= num_examples)
		samples = random.sample(samples, num_examples)
		
		for sample in samples:
			result.append(self[sample])

		return result
