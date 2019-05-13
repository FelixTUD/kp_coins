from keras.models import Model
from keras.layers import Input, LSTM, Dense, CuDNNLSTM
from keras.utils import Sequence

import os
import h5py
import numpy as np
import argparse

class TrainingGenerator(Sequence):
	def __init__(self, batch_size, clip_length, shrink, file_path, coin_class_override=None):
		self.batch_size = batch_size
		
		self.file_path = file_path
		self.clip_length = clip_length
		self.shrink = shrink

		self.data_file = h5py.File(self.file_path, "r")

		if not coin_class_override:
			self.coins = [int(x) for x in self.data_file.keys()]
		else:
			print("Coin class override, sampling from classes: {}".format(coin_class_override))
			self.coins = coin_class_override

		self.n_classes = len(self.coins)

	def get_random_example(self):
		coin_choice = np.random.choice(self.coins, 1)[0]
		gain_choice = np.random.choice(["g8", "g16"], 1)[0]

		possible_examples = list(self.data_file[str(coin_choice)][gain_choice].keys())
		example_choice = np.random.choice(possible_examples, 1)[0]

		example_data = self.data_file[str(coin_choice)][gain_choice][example_choice]["values"][:][:self.clip_length:self.shrink]

		example_max = np.max(example_data)
		exmaple_min = np.min(example_data)
		example_data = (example_data - exmaple_min) / (example_max - exmaple_min)

		return example_data.reshape(example_data.size, 1)

	def __getitem__(self, idx):
		input_result_batch = np.empty((self.batch_size, self.clip_length//self.shrink+1, 1))
		teacher_result_batch = np.empty((self.batch_size, self.clip_length//self.shrink+1, 1))
		output_result_batch = np.empty((self.batch_size, self.clip_length//self.shrink+1, 1))

		for batch_i in range(self.batch_size):
			random_example = self.get_random_example()
			input_result_batch[batch_i] = random_example
			teacher_input = np.flip(random_example, 0)[:-1]
			teacher_input = np.insert(teacher_input, 0, [-1])
			teacher_result_batch[batch_i] = teacher_input.reshape(teacher_input.size, 1)
			output_result_batch[batch_i] = np.flip(random_example, 0)

		return [input_result_batch, teacher_result_batch], output_result_batch

	def __len__(self):
		return 200


def generate_model(hidden_dim, feature_size):
	# Define an input sequence and process it.
	encoder_inputs = Input(shape=(None, feature_size))
	encoder = CuDNNLSTM(hidden_dim, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	# We discard `encoder_outputs` and only keep the states.
	encoder_states = [state_h, state_c]

	# Set up the decoder, using `encoder_states` as initial state.
	decoder_inputs = Input(shape=(None, feature_size))
	# We set up our decoder to return full output sequences,
	# and to return internal states as well. We don't use the 
	# return states in the training model, but we will use them in inference.
	decoder_lstm = CuDNNLSTM(hidden_dim, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
	                                     initial_state=encoder_states)
	decoder_dense = Dense(feature_size, activation='sigmoid')
	decoder_outputs = decoder_dense(decoder_outputs)

	# Define the model that will turn
	# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

	model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

	return model

def main(args):
	model = generate_model(args.hidden_size, 1)

	training_generator = TrainingGenerator(batch_size=args.batch_size, clip_length=700000, shrink=args.shrink, file_path=args.file_path, coin_class_override=[5, 100])

	model.fit_generator(training_generator, steps_per_epoch=200, epochs=args.epochs, shuffle=True)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	parser.add_argument("-b", "--batch_size", type=int, default=20, help="Batch size. Default 1")
	parser.add_argument("-p", "--file_path", type=str, default=None, help="Path to dataset file")
	parser.add_argument("-s", "--shrink", type=int, help="Shrinking factor. Selects data every s steps from input.")
	parser.add_argument("-hs", "--hidden_size", type=int, help="Size of LSTM/GRU hidden layer.")
	parser.add_argument("-e", "--epochs", type=int, default=50, help="Number of epochs")
	parser.add_argument("--save", type=str, default=None, help="Specify save folder for weight files. Default: None")
	parser.add_argument("-w", "--weights", type=str, default=None, help="Model weights file. Only used for 'tsne' mode. Default: None")

	args = parser.parse_args()

	main(args)
