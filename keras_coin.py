from keras.models import Model
from keras.layers import Input, LSTM, Dense, CuDNNLSTM, Add
from keras.utils import Sequence, to_categorical
from keras.callbacks import TensorBoard
from keras import backend as K

import os
import h5py
import numpy as np
import argparse
import socket
import random
from datetime import datetime
from collections import defaultdict

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

		self.indexes = defaultdict(list)
		self.num_training_examples = 0

		for coin in self.coins:
			for gain in ["g8", "g16"]:
				example_indexes = list(map(lambda x: (x, gain), list(self.data_file[str(coin)][gain].keys())))
				random.shuffle(example_indexes)
				self.indexes[int(coin)] = example_indexes

				self.num_training_examples += len(example_indexes)

		print("Indexed {} training examples".format(self.num_training_examples))

	def get_random_example(self):
		coin_choice = np.random.choice(list(self.indexes.keys()), 1)[0]

		possible_examples = self.indexes[coin_choice]
		example_choice = random.sample(possible_examples, 1)[0]

		example_data = self.data_file[str(coin_choice)][example_choice[1]][example_choice[0]]["values"][:][:self.clip_length:self.shrink]

		example_max = np.max(example_data)
		exmaple_min = np.min(example_data)
		example_data = (example_data - exmaple_min) / (example_max - exmaple_min)

		return to_categorical(np.where(self.coins==coin_choice)[0][0], self.n_classes), example_data.reshape(example_data.size, 1)

	def __getitem__(self, idx):
		input_result_batch = np.empty((self.batch_size, self.clip_length//self.shrink+1, 1))
		teacher_result_batch = np.empty((self.batch_size, self.clip_length//self.shrink+1, 1))
		output_result_batch = np.empty((self.batch_size, self.clip_length//self.shrink+1, 1))
		classification_result_batch = np.empty((self.batch_size, self.n_classes))

		for batch_i in range(self.batch_size):
			classification, random_example = self.get_random_example()
			input_result_batch[batch_i] = random_example
			teacher_input = np.flip(random_example, 0)[:-1]
			teacher_input = np.insert(teacher_input, 0, [-1])
			teacher_result_batch[batch_i] = teacher_input.reshape(teacher_input.size, 1)
			output_result_batch[batch_i] = np.flip(random_example, 0)
			classification_result_batch[batch_i] = classification

		return [input_result_batch, teacher_result_batch], {"decoder_out" : output_result_batch, "classification_out": classification_result_batch}

	def __len__(self):
		return self.num_training_examples // self.batch_size


def generate_model(hidden_dim, feature_size, n_classes):
	gpu_available = len(list(K.tensorflow_backend._get_available_gpus())) > 0

	# Define an input sequence and process it.
	encoder_inputs = Input(shape=(None, feature_size))
	encoder = None
	if gpu_available:
		encoder = CuDNNLSTM(hidden_dim, return_state=True)
	else:
		encoder = LSTM(hidden_dim, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	# We discard `encoder_outputs` and only keep the states.
	encoder_states = [state_h, state_c]

	# Set up the decoder, using `encoder_states` as initial state.
	decoder_inputs = Input(shape=(None, feature_size))
	# We set up our decoder to return full output sequences,
	# and to return internal states as well. We don't use the 
	# return states in the training model, but we will use them in inference.
	decoder_lstm = None
	if gpu_available:
		decoder_lstm = CuDNNLSTM(hidden_dim, return_sequences=True, return_state=True)
	else:
		decoder_lstm = LSTM(hidden_dim, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
	                                     initial_state=encoder_states)
	decoder_dense = Dense(feature_size, activation='sigmoid', name="decoder_out")
	decoder_outputs = decoder_dense(decoder_outputs)

	classificator_fc_h_1 = Dense(hidden_dim, activation="sigmoid")(state_h)
	classificator_fc_c_1 = Dense(hidden_dim, activation="sigmoid")(state_c)
	classificator_combine = Add()([classificator_fc_h_1, classificator_fc_c_1])
	classificator_fc_out = Dense(n_classes, activation="softmax", name="classification_out")(classificator_combine)

	# Define the model that will turn
	# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
	model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=[decoder_outputs, classificator_fc_out])

	losses = {
		"classification_out": "categorical_crossentropy",
		"decoder_out": "mse"
	}

	metrics = {
		"classification_out": ["categorical_accuracy"],
		"decoder_out": []
	}

	model.compile(optimizer="adam", loss=losses, metrics=metrics)

	print(model.summary())

	return model

def get_comment_string(args):
	comment = "::d_" #if args.debug else ""
	comment += "b{}_".format(args.batch_size)
	comment += "hs{}_".format(args.hidden_size)
	comment += "lstm_" #if args.use_lstm else "gru_"
	comment += "s{}_".format(args.shrink)
	comment += "e{}".format(args.epochs)
	return comment

def get_log_dir(comment):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', current_time + '_' + socket.gethostname() + comment)
    return str(log_dir)

def main(args):
	model = generate_model(args.hidden_size, 1, n_classes=2)

	training_generator = TrainingGenerator(batch_size=args.batch_size, clip_length=700000, shrink=args.shrink, file_path=args.file_path, coin_class_override=[5, 100])

	callbacks = []
	callbacks += [TensorBoard(log_dir=get_log_dir(get_comment_string(args)), batch_size=args.batch_size)]

	model.fit_generator(training_generator, steps_per_epoch=len(training_generator), epochs=args.epochs, shuffle=True, callbacks=callbacks)

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
