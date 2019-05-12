import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import multiprocessing
import pandas 
import sys
import argparse
import time
import os
import matplotlib.pyplot as plt

from dataset import CoinDatasetLoader, CoinDataset
from model import Autoencoder

def custom_mse_loss(y_pred, y_true):
	return ((y_true-y_pred)**2).sum(1).mean()

def calc_acc(input, target):
	#input = nn.functional.softmax(input, 1)
	#print(torch.argmax(input, 1))
	return (torch.argmax(input, 1) == target).sum().item() / input.shape[0]

global_step = 0

def train(model, dataloader, optimizer, loss_fn, writer=None):
	global global_step
	model = model.train()
	num_steps = len(dataloader)

	loss_history = np.empty(num_steps)
	loss_history_cel = np.empty(num_steps)
	acc_history = np.empty(num_steps)
	loss_cel = nn.CrossEntropyLoss()

	for i_batch, sample_batched in enumerate(dataloader):
		print("{}/{}".format(i_batch + 1, num_steps), end="\r")
		input_tensor, reversed_input, teacher_input, output = sample_batched["input"], sample_batched["reversed_input"], sample_batched["teacher_input"], sample_batched["label"]

		model.set_decoder_mode(True)
		predicted_sequence = model(input=input_tensor, teacher_input=teacher_input)

		loss = loss_fn(predicted_sequence, reversed_input)
		loss_history[i_batch] = loss.item()

		if writer:
			writer.add_scalar("raw/loss/reconstruction", global_step=global_step, scalar_value=loss.item())

		optimizer[0].zero_grad()
		loss.backward()
		optimizer[0].step()

		del loss # Necessary?
	
		model.set_decoder_mode(False)
		predicted_category = model(input=input_tensor, teacher_input=None)

		loss = loss_cel(input=predicted_category, target=output)
		loss_history_cel[i_batch] = loss.item()
		acc = calc_acc(input=predicted_category, target=output)
		acc_history[i_batch] = acc

		if writer:
			writer.add_scalar("raw/loss/categorization", global_step=global_step, scalar_value=loss.item())
			writer.add_scalar("raw/acc/categorization", global_step=global_step, scalar_value=acc)

		optimizer[1].zero_grad()
		loss.backward()
		optimizer[1].step()

		del loss # Necessary?

		global_step += 1

	if writer:
		writer.add_scalar("per_epoch/loss/categorization", global_step=global_step // num_steps, scalar_value=loss_history_cel.mean())
		writer.add_scalar("per_epoch/loss/reconstruction", global_step=global_step // num_steps, scalar_value=loss_history.mean())
		writer.add_scalar("per_epoch/acc/categorization", global_step=global_step // num_steps, scalar_value=acc_history.mean())
			
	return {"loss_mean": loss_history.mean(), "loss_cel_mean": loss_history_cel.mean(), "accuracy": acc_history.mean()}

def evaluate(epoch, model, dataloader, loss_fn, start_of_sequence=-1, writer=None):
	model = model.eval()
	model.set_eval_mode(True)
	model.set_decoder_mode(False)

	num_steps = len(dataloader)

	loss_history = np.empty(num_steps)
	loss_history_cel = np.empty(num_steps)
	acc_history = np.empty(num_steps)
	loss_cel = nn.CrossEntropyLoss()

	# plot_dir = os.path.join(base_path, "plots/" + str(epoch))
	# os.makedirs(plot_dir, exist_ok=True)

	for i_batch, sample_batched in enumerate(dataloader):
		print("{}/{}".format(i_batch + 1, num_steps), end="\r")
		input_tensor, reversed_input, _, output = sample_batched["input"], sample_batched["reversed_input"], sample_batched["teacher_input"], sample_batched["label"]

		# iterative_teacher_input = torch.empty(input_tensor.shape).to(input_tensor.device)
		# iterative_teacher_input[:, 0,:] = -1

		# predicted_sequence = model(input=input_tensor, teacher_input=iterative_teacher_input)

		# for i in range(input_tensor.shape[0]):
		# 	np_input = input_tensor[i].cpu().numpy().reshape(input_tensor.shape[1])
		# 	np_predicted = np.flip(predicted_sequence[i].detach().cpu().numpy().reshape(predicted_sequence.shape[1]))

		# 	plt.plot(np_input, label="input")
		# 	plt.plot(np_predicted, label="predicted")
		# 	plt.savefig(os.path.join(plot_dir, str(i_batch * num_steps + i) + ".png"), format="png")
		# 	plt.clf()

		# loss = loss_fn(predicted_sequence, reversed_input)

		# loss_history[i_batch] = loss.item()

		predicted_category = model(input=input_tensor, teacher_input=None)

		loss = loss_cel(input=predicted_category, target=output)
		loss_history_cel[i_batch] = loss.item()
		acc = calc_acc(input=predicted_category, target=output)
		acc_history[i_batch] = acc

		if writer:
			# writer.add_scalar("val_raw/loss/categorization", global_step=global_step, scalar_value=loss.item())
			writer.add_scalar("val_raw/acc/categorization", global_step=global_step, scalar_value=acc)

		del loss # Necessary?

	if writer:
		writer.add_scalar("val_per_epoch/loss/categorization", global_step=global_step // num_steps, scalar_value=loss_history_cel.mean())
		# writer.add_scalar("val_per_epoch/loss/reconstruction", global_step=global_step // num_steps, scalar_value=loss_history.mean())
		writer.add_scalar("val_per_epoch/acc/categorization", global_step=global_step // num_steps, scalar_value=acc_history.mean())

	model.set_eval_mode(False)
	return {"loss_mean": 0, "loss_cel_mean": loss_history_cel.mean(), "accuracy": acc_history.mean()}

def get_dict_string(d, prefix=""):
	result = prefix
	for k, v in d.items():
		result += " {}: {:.4f},".format(k, v)

	return result[:-1]

def get_comment_string(args):
	comment = "::d_" if args.debug else ""
	comment += "b{}_".format(args.batch_size)
	comment += "hs{}_".format(args.hidden_size)
	comment += "lstm_" if args.use_lstm else "gru_"
	comment += "s{}_".format(args.shrink)
	comment += "e{}".format(args.epochs)
	return comment

def main(args):
	print("CPU count: {}".format(args.cpu_count))

	torch.set_num_threads(max(1, args.cpu_count))
	cuda_available = torch.cuda.is_available()

	writer = SummaryWriter(comment=get_comment_string(args))
	model_save_dir_name = writer.log_dir.split("/")[-1]
	model_save_path = None

	if args.save:
		model_save_path = os.path.join(args.save, model_save_dir_name)
		os.makedirs(model_save_path)

	dataset_loader = CoinDatasetLoader(path_to_hdf5=os.path.join(args.path, "coin_data/data.hdf5"), shrink=args.shrink, validation_split=0.1, test_split=0.1, args=args)

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

	training_dataloader = DataLoader(training_dataset, batch_size=training_batch_size, shuffle=True, num_workers=0, drop_last=True)	
	validation_dataloader = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=0, drop_last=(validation_batch_size > 1))
	test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0, drop_last=(test_batch_size > 1))

	training_dataset_length = int(len(training_dataset) / training_batch_size)
	validation_dataset_length = int(len(validation_dataset) / validation_batch_size)
	test_dataset_length = int(len(test_dataset) / test_batch_size)

	model = Autoencoder(hidden_dim=args.hidden_size, feature_dim=1, use_lstm=args.use_lstm, activation_function=nn.Sigmoid(), args=args)
	if cuda_available:
		print("Moving model to gpu...")
		model = model.cuda()

	# model = nn.DataParallel(model, device_ids=[0, 1, 2])

	opti = [optim.Adam(model.get_autoencoder_param()), optim.Adam(model.get_predictor_param())]
	loss_fn = custom_mse_loss

	num_epochs = args.epochs

	print(model)
	print("Num parameters: {}".format(model.num_parameters()))

	training_loss_history = np.empty(training_dataset_length)
	validation_loss_history = np.empty(validation_dataset_length)

	num_epochs_no_improvements = 0
	best_val_loss = np.inf
	no_improvements_patience = 5
	no_improvements_min_epochs = 10

	base_path = os.path.join(args.path, args.identifier) 

	log_file_dir = os.path.dirname(args.log_file)
	if log_file_dir:
		os.makedirs(log_file_dir, exist_ok=True)

	if args.mode == "train":
		with open(os.path.join(args.path, args.log_file), "w") as log_file:
			log_file.write("epoch, loss, val_loss\n")

			for current_epoch in range(num_epochs):

				start_time = time.time()
				train_history = train(model=model, dataloader=training_dataloader, optimizer=opti, loss_fn=custom_mse_loss, writer=writer)
				end_time = time.time()

				print("Elapsed training time: {:.2f} seconds".format(end_time - start_time))
				
				start_time = time.time()
				validation_history = evaluate(epoch=current_epoch+1, model=model, dataloader=validation_dataloader, loss_fn=custom_mse_loss, writer=writer)
				end_time = time.time()

				print("Elapsed validation time: {:.2f} seconds".format(end_time - start_time))

				# print("Epoch {}/{}: loss: {:.5f}".format(current_epoch + 1, num_epochs, validation_history["loss_mean"]))
				# log_file.write("{}, {}\n".format(current_epoch + 1, validation_history["loss_mean"]))
				#print("Epoch {}/{}: loss: {:.5f}, val_loss: {:.5f}, cel_loss: {:.5f}".format(current_epoch + 1, num_epochs, train_history["loss_mean"], validation_history["loss_mean"], train_history["loss_cel_mean"]))
				#log_file.write("{}, {}, {}\n".format(current_epoch + 1, train_history["loss_mean"], validation_history["loss_mean"]))
				# print("Epoch {}/{}: loss: {:.5f}, cel_loss: {:.5f}, accuracy:  {:.5f}".format(current_epoch + 1, num_epochs, train_history["loss_mean"], train_history["loss_cel_mean"], train_history["accuracy"]))
				print("Epoch {}/{}:".format(current_epoch + 1, num_epochs))
				print(get_dict_string(train_history, "train: "))
				print(get_dict_string(validation_history, "val: "))
				print("---")
				# log_file.write("{}, {}\n".format(current_epoch + 1, train_history["loss_mean"]))
				# log_file.flush()

				if args.save:
					torch.save(model.state_dict(), os.path.join(model_save_path, "{:04d}.weights".format(current_epoch + 1)))

	if args.mode == "tsne":
		from sklearn.manifold import TSNE
		assert (args.weights), "No weights file specified!"

		device = torch.device('cpu')
		model.load_state_dict(torch.load(args.weights, map_location=device))
		model.eval()
		# Use training examples for now to test
		num_examples_per_class = 25
		coins = [5, 100]
		colors = ['b', 'r']

		plot_colors = []
		[plot_colors.extend(x*num_examples_per_class) for x in colors]

		i = 0
		all_encodings = torch.empty(num_examples_per_class*len(coins), args.hidden_size)
		for coin in coins:
			coin_data = training_dataset.get_data_for_coin_type(coin=coin, num_examples=num_examples_per_class)

			for data in coin_data:
				print("\rCoin {}: {}/{}".format(coin, (i+1) % (num_examples_per_class + 1), num_examples_per_class), end="")
				input_tensor = data["input"]
				encoded_input = model(input=input_tensor.view(1, input_tensor.shape[0], input_tensor.shape[1]), return_hidden=True)

				all_encodings[i] = encoded_input[0]
				i += 1
			print("")
		embedded = TSNE(n_components=2).fit_transform(all_encodings.numpy())

		plt.scatter(embedded[:,0], embedded[:,1], c=plot_colors, alpha=0.5)
		plt.show()

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

	writer.close()

if __name__ == "__main__":
	torch.multiprocessing.set_start_method("spawn")

	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--mode", type=str, default="train", help="Mode of the script. Can be either 'train', 'tsne' or infer'. Default 'train'")
	parser.add_argument("-c", "--cpu_count", type=int, default=0, help="Number of cpus to use. Default 0")
	parser.add_argument("-b", "--batch_size", type=int, default=20, help="Batch size. Default 1")
	parser.add_argument("-lstm", "--use_lstm", type=bool, default=True, help="Use lstm or gru. Default True = use lstm")
	parser.add_argument("-l", "--log_file", type=str, default="metrics.csv", help="CSV logfile. Creates path if it does not exist. Default 'metrics.csv'")
	parser.add_argument("-p", "--path", type=str, default="./", help="Path to working directory, used as base dataset path and base log file path. Default ./")
	parser.add_argument("-s", "--shrink", type=int, help="Shrinking factor. Selects data every s steps from input.")
	parser.add_argument("-hs", "--hidden_size", type=int, help="Size of LSTM/GRU hidden layer.")
	parser.add_argument("-id", "--identifier", type=str, help="Unique identifier for the current run.")
	parser.add_argument("-d", "--debug", action="store_true")
	parser.add_argument("-e", "--epochs", type=int, default=50, help="Number of epochs")
	parser.add_argument("--save", type=str, default=None, help="Specify save folder for weight files. Default: None")
	parser.add_argument("-w", "--weights", type=str, default=None, help="Model weights file. Only used for 'tsne' mode. Default: None")

	args = parser.parse_args()

	main(args)
