from keras.models import Model
from keras.layers import Input, LSTM, Dense, CuDNNLSTM, Add, Concatenate, GRU, CuDNNGRU, TimeDistributed
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
import librosa as rosa

class TrainingGenerator(Sequence):
	def __init__(self, batch_size, clip_length, shrink, file_path, top_db, coin_class_override=None):
		self.batch_size = batch_size
		
		self.file_path = file_path
		self.clip_length = clip_length
		self.shrink = shrink
		self.top_db = top_db

		self.data_file = h5py.File(self.file_path, "r")

		if not coin_class_override:
			self.coins = [int(x) for x in self.data_file.keys()]
		else:
			print("Coin class override, sampling from classes: {}".format(coin_class_override))
			self.coins = coin_class_override

		self.n_classes = len(self.coins)

		self.indexes = defaultdict(list)
		self.num_training_examples = 0

		self.data = []
		self.longest_audio = 0

		i = 0
		for coin in self.coins:
			for gain in ["g8", "g16"]:
				example_indexes = list(self.data_file[str(coin)][gain].keys())

				for num, index in enumerate(example_indexes[:40]):
					print("\rLoading {}, {} ({}/{})".format(coin, gain, num + 1, len(example_indexes)), end="")
					audio_data = self.data_file[str(coin)][gain][str(index)]["values"][:]
					audio_data = rosa.util.normalize(audio_data)
					clipped_audio = rosa.effects.trim(audio_data, top_db=self.top_db)[0][:self.clip_length:self.shrink]

					# clipped_audio_max = np.max(clipped_audio)
					# clipped_audio_min = np.min(clipped_audio)
					# scaled_audio = (clipped_audio - clipped_audio_min) / (clipped_audio_max - clipped_audio_min)

					self.longest_audio = max(clipped_audio.size, self.longest_audio)

					self.data.append((to_categorical(np.where(np.array(self.coins)==coin)[0][0], len(self.coins)), clipped_audio.reshape(clipped_audio.size, 1)))

					self.num_training_examples += 1
				print("")

		print("Padding all audio to length: {}".format(self.longest_audio))
		self.data = list(map(self.pad, self.data))
		random.shuffle(self.data)
		print("Indexed {} training examples".format(self.num_training_examples))

	def on_epoch_end(self):
		random.shuffle(self.data)

	def pad(self, data_example):
		coin = data_example[0]
		audio_data = data_example[1]
		missing_values = self.longest_audio - audio_data.size
		if missing_values > 0:
			audio_data = audio_data.reshape(audio_data.size)
			audio_data = np.pad(audio_data, (0, missing_values), 'constant', constant_values=(0))
			audio_data = audio_data.reshape(audio_data.size, 1)
		return (coin, audio_data)

	def get_random_example(self):
		coin_choice = np.random.choice(list(self.indexes.keys()), 1)[0]

		possible_examples = self.indexes[coin_choice]
		example_choice = random.sample(possible_examples, 1)[0]

		example_data = self.data_file[str(coin_choice)][example_choice[1]][example_choice[0]]["values"][:]#[:self.clip_length:self.shrink]
		example_data = rosa.util.normalize(example_data)
		clipped = rosa.effects.trim(example_data, top_db=self.top_db)[0][:self.clip_length:self.shrink]

		return to_categorical(np.where(self.coins==coin_choice)[0][0], self.n_classes), example_data.reshape(example_data.size, 1)

	def __getitem__(self, idx):
		input_result_batch = np.empty((self.batch_size, self.longest_audio, 1))
		teacher_result_batch = np.empty((self.batch_size, self.longest_audio, 1))
		output_result_batch = np.empty((self.batch_size, self.longest_audio, 1))
		classification_result_batch = np.empty((self.batch_size, self.n_classes))

		for batch_i in range(self.batch_size):
			classification, random_example = self.data[idx * self.batch_size + batch_i]
			input_result_batch[batch_i] = random_example
			teacher_input = np.flip(random_example, 0)[:-1]
			teacher_input = np.insert(teacher_input, 0, [-1])
			teacher_result_batch[batch_i] = teacher_input.reshape(teacher_input.size, 1)
			output_result_batch[batch_i] = np.flip(random_example, 0)
			classification_result_batch[batch_i] = classification

		return [input_result_batch, teacher_result_batch], {"decoder_out" : output_result_batch, "classification_out": classification_result_batch}

	def __len__(self):
		return self.num_training_examples // self.batch_size

def custom_mse_loss(y_true, y_pred):
	return K.mean(K.sum(K.square(y_true - y_pred), axis=1), axis=0)

def generate_model(args, feature_size, n_classes):
	gpu_available = len(list(K.tensorflow_backend._get_available_gpus())) > 0

	# Define an input sequence and process it.
	encoder_inputs = Input(shape=(None, feature_size))
	encoder = None
	if gpu_available:
		if not args.use_gru:
			encoder = CuDNNLSTM(args.hidden_size, return_state=True)
		else:
			encoder = CuDNNGRU(args.hidden_size, return_state=True)
	else:
		if not args.use_gru:
			encoder = LSTM(args.hidden_size, return_state=True)
		else:
			encoder = GRU(args.hidden_size, return_state=True)

	last_out, state_h, state_c = None, None, None
	if not args.use_gru:
		last_out, state_h, state_c = encoder(encoder_inputs)
	else:
		last_out, state_h = encoder(encoder_inputs)

	# We discard `encoder_outputs` and only keep the states.
	encoder_states = None
	if not args.use_gru:
		encoder_states = [state_h, state_c]
	else:
		encoder_states = state_h

	# Set up the decoder, using `encoder_states` as initial state.
	decoder_inputs = Input(shape=(None, feature_size))
	# We set up our decoder to return full output sequences,
	# and to return internal states as well. We don't use the 
	# return states in the training model, but we will use them in inference.
	decoder = None
	if gpu_available:
		if not args.use_gru:
			decoder = CuDNNLSTM(args.hidden_size, return_sequences=True)
		else:
			decoder = CuDNNGRU(args.hidden_size, return_sequences=True)
	else:
		if not args.use_gru:
			decoder = LSTM(args.hidden_size, return_sequences=True)
		else:
			decoder = GRU(args.hidden_size, return_sequences=True)
	decoder_outputs = decoder(decoder_inputs,
	                                initial_state=encoder_states)
	decoder_dense = TimeDistributed(Dense(feature_size, activation='tanh'), name="decoder_out")
	decoder_outputs = decoder_dense(decoder_outputs)

	if not args.use_gru:
		classificator_fc_h_1 = Dense(args.hidden_size//2, activation="relu")(last_out)
		# classificator_fc_c_1 = Dense(args.hidden_size//2, activation="relu")(state_c)
		# classificator_combine = Concatenate()([classificator_fc_h_1, classificator_fc_c_1])
		classificator_fc_out = Dense(n_classes, activation="softmax", name="classification_out")(classificator_fc_h_1)
	else:
		classificator_fc_h = Dense(args.hidden_size//2, activation="relu")(last_out)
		classificator_fc_out = Dense(n_classes, activation="softmax", name="classification_out")(classificator_fc_h)

	# Define the model that will turn
	# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
	model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=[decoder_outputs, classificator_fc_out]) 

	losses = {
		"classification_out": "categorical_crossentropy",
		"decoder_out": custom_mse_loss
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
	comment += "gru_" if args.use_gru else "lstm_"
	comment += "s{}_".format(args.shrink)
	comment += "e{}".format(args.epochs)
	return comment

def get_log_dir(comment):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', current_time + '_' + socket.gethostname() + comment)
    return str(log_dir)

def main(args):
	model = generate_model(args, 1, n_classes=2)

	training_generator = TrainingGenerator(batch_size=args.batch_size, clip_length=700000, shrink=args.shrink, file_path=args.file_path, top_db=args.top_db, coin_class_override=[5, 100])

	callbacks = []
	callbacks += [TensorBoard(log_dir=get_log_dir(get_comment_string(args)), batch_size=args.batch_size)]

	model.fit_generator(training_generator, steps_per_epoch=len(training_generator), epochs=args.epochs, shuffle=True, callbacks=callbacks)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	parser.add_argument("-b", "--batch_size", type=int, default=20, help="Batch size. Default 1")
	parser.add_argument("-p", "--file_path", type=str, default=None, help="Path to dataset file")
	parser.add_argument("-s", "--shrink", type=int, help="Shrinking factor. Selects data every s steps from input.")
	parser.add_argument("-gru", "--use_gru", action="store_true", help="Use  gru. If not set, uses LSTMs.")
	parser.add_argument("-hs", "--hidden_size", type=int, help="Size of LSTM/GRU hidden layer.")
	parser.add_argument("-e", "--epochs", type=int, default=50, help="Number of epochs")
	parser.add_argument("--save", type=str, default=None, help="Specify save folder for weight files. Default: None")
	parser.add_argument("-w", "--weights", type=str, default=None, help="Model weights file. Only used for 'tsne' mode. Default: None")
	parser.add_argument("-db", "--top_db", type=int, default=10, help="The threshold (in decibels) below reference to consider as silence")

	args = parser.parse_args()

	main(args)
