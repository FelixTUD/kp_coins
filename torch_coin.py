import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np
import multiprocessing
import pandas 
import sys
import argparse
import time
import os

from dataset import CoinDatasetLoader, CoinDataset
from model import Autoencoder

def custom_mse_loss(y_pred, y_true):
	return ((y_true-y_pred)**2).sum(1).mean()

def main(args):
	print("CPU count: {}".format(args.cpu_count))

	torch.set_num_threads(args.cpu_count)
	cuda_available = torch.cuda.is_available()

	dataset_loader = CoinDatasetLoader(path_to_hdf5=os.path.join(args.path, "coin_data/data.hdf5"), validation_split=0.1, test_split=0.1)

	print("Loading training set")
	training_dataset = dataset_loader.get_dataset("training")
	print("Loading validation set")
	validation_dataset = dataset_loader.get_dataset("validation")
	print("Loading test set")
	test_dataset = dataset_loader.get_dataset("test")

	print("Training dataset length: {}".format(len(training_dataset)))
	print("Validation dataset length: {}".format(len(validation_dataset)))
	print("Test dataset length: {}".format(len(test_dataset)))

	training_batch_size = args.batch_size
	validation_batch_size = args.batch_size
	test_batch_size = args.batch_size

	training_dataloader = DataLoader(training_dataset, batch_size=training_batch_size, shuffle=True, num_workers=args.cpu_count, drop_last=True)	
	validation_dataloader = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=args.cpu_count, drop_last=(validation_batch_size > 1))
	test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=args.cpu_count, drop_last=(test_batch_size > 1))

	training_dataset_length = int(len(training_dataset) / training_batch_size)
	validation_dataset_length = int(len(validation_dataset) / validation_batch_size)
	test_dataset_length = int(len(test_dataset) / test_batch_size)

	model = Autoencoder(hidden_dim=2**6, feature_dim=1, use_lstm=args.use_lstm, activation_function=nn.Sigmoid())
	if cuda_available:
		print("Moving model to gpu...")
		model = model.cuda()

	# model = nn.DataParallel(model, device_ids=[0, 1, 2])

	opti = optim.Adam(model.parameters())
	loss_fn = custom_mse_loss

	num_epochs = 50

	print(model)
	print("Num parameters: {}".format(model.num_parameters()))

	training_loss_history = np.empty(training_dataset_length)
	validation_loss_history = np.empty(validation_dataset_length)

	num_epochs_no_improvements = 0
	best_val_loss = np.inf
	no_improvements_patience = 5
	no_improvements_min_epochs = 10

	log_file_dir = os.path.dirname(args.log_file)
	if log_file_dir:
		os.makedirs(log_file_dir, exist_ok=True)

	if args.mode == "train":
		with open(os.path.join(args.path, args.log_file), "w") as log_file:

			for current_epoch in range(num_epochs):
				training_loss_history[:] = 0
				validation_loss_history[:] = 0

				start_time = time.time()

				for i_batch, sample_batched in enumerate(training_dataloader):
					print("{}/{}".format(i_batch + 1, training_dataset_length), end="\r")
					input_tensor, reversed_input, teacher_input, output = sample_batched["input"], sample_batched["reversed_input"], sample_batched["teacher_input"], sample_batched["label"]

					# if cuda_available:
					# 	input_tensor = input_tensor.cuda()

					# reversed_input = torch.flip(input_tensor, dims=(1, 2))
					# teacher_input = torch.cat((start_of_sequence, input_tensor[:,:-1,:]), 1)

					predicted_sequence = model(input=input_tensor, teacher_input=teacher_input)

					loss = loss_fn(predicted_sequence, reversed_input)

					training_loss_history[i_batch] = loss.item()

					opti.zero_grad()

					loss.backward()

					opti.step()

					del loss # Necessary?

				end_time = time.time()

				print("Elapsed time: {:2f} seconds".format(end_time - start_time))

				# for val_i_batch, val_sample_batch in enumerate(validation_dataloader):

				# 	input_tensor, output = val_sample_batch["input"], val_sample_batch["output"]
				# 	teacher_input = input_tensor[:,-1,:].reshape(validation_batch_size, 1, 1)

				# 	# Predict iteratively

				# 	for _ in range(num_to_predict):

				# 		partial_predicted_sequence = model(input=input_tensor, teacher_input=teacher_input)

				# 		teacher_input = torch.cat((teacher_input, partial_predicted_sequence[:,-1,:].reshape(validation_batch_size, 1, 1)), 1)

				# 	loss = mse_loss(partial_predicted_sequence, output)

				# 	validation_loss_history[val_i_batch] = loss.item()

				# val_loss = validation_loss_history.mean()

				# if best_val_loss < val_loss:
				# 	num_epochs_no_improvements += 1
				# else:
				# 	num_epochs_no_improvements = 0
				# 	torch.save(model.state_dict(), "rae_teacher_forcing_weights.pt")

				# best_val_loss = np.minimum(best_val_loss, val_loss)

				training_epoch_loss = training_loss_history.mean()
				print("Epoch {}/{}: loss: {:5f}".format(current_epoch + 1, num_epochs, training_epoch_loss))
				log_file.write("{}, {}\n".format(current_epoch + 1, training_epoch_loss))
				log_file.flush()

				# print("Epoch {}/{}: loss: {:5f}, val_loss: {:5f}".format(current_epoch + 1, num_epochs, training_loss_history.mean(), val_loss))

				# if num_epochs_no_improvements == no_improvements_patience and no_improvements_min_epochs < current_epoch:
				# 	print("No imprevements in val loss for 3 epochs. Aborting training.")
				# 	break

	if args.mode == "infer":
		model.load_state_dict(torch.load("rae_teacher_forcing_weights.pt"))

		gt_val = []
		pred_val = []

		for i, val_sample_batch in enumerate(validation_dataset):

			if i % num_to_predict == 0:

				input_tensor, output, (de_normalize_min, de_normalize_prod)  = val_sample_batch["input"], val_sample_batch["output"], val_sample_batch["de_normalize"]
				input_tensor = input_tensor.reshape(1, input_tensor.shape[0], input_tensor.shape[1])
				teacher_input = input_tensor[:,-1,:].reshape(1, 1, 1)

				gt_val += list((((output.cpu().numpy().reshape(num_to_predict)) * de_normalize_prod) + de_normalize_min))

				# Predict iteratively

				for _ in range(num_to_predict):

					partial_predicted_sequence = model(input=input_tensor, teacher_input=teacher_input)

					teacher_input = torch.cat((teacher_input, partial_predicted_sequence[:,-1,:].reshape(1, 1, 1)), 1)

				pred_val += list(((partial_predicted_sequence[0].cpu().detach().numpy().reshape(num_to_predict)) * de_normalize_prod) + de_normalize_min)

		plt.plot(np.arange(len(gt_val)), gt_val, label="gt")
		plt.plot(np.arange(len(pred_val)), pred_val, label="pred")
		plt.legend()
		plt.savefig("val_pred_plot.pdf", format="pdf")
		plt.show()

if __name__ == "__main__":
	torch.multiprocessing.set_start_method("spawn")

	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--mode", type=str, default="train", help="Mode of the script. Can be either 'train' or 'infer'. Default 'train'")
	parser.add_argument("-c", "--cpu_count", type=int, default=1, help="Number of cpus to use. Default 1")
	parser.add_argument("-b", "--batch_size", type=int, default=1, help="Batch size. Default 1")
	parser.add_argument("-lstm", "--use_lstm", type=bool, default=True, help="Use lstm or gru. Default True = use lstm")
	parser.add_argument("-l", "--log_file", type=str, default="metrics.csv", help="CSV logfile. Creates path if it does not exist. Default 'metrics.csv'")
	parser.add_argument("-p", "--path", type=str, default="./", help="Path to working directory, used as base dataset path and base log file path. Default ./")

	args = parser.parse_args()

	main(args)