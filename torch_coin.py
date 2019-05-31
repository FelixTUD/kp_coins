import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_packed_sequence

import numpy as np
import multiprocessing
import pandas 
import sys
import argparse
import time
import os
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from dataset import CoinDatasetLoader, CoinDataset, NewCoinDataset, Collator
from model import Autoencoder, VariationalAutoencoder

def custom_mse_loss(y_pred, y_true):
	return ((y_true-y_pred)**2).sum(1).mean()

def calc_acc(input, target):
	return (torch.argmax(input, 1) == target).sum().item() / input.shape[0]

def kl_loss(mu, logvar):
	return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def calculate_loss_non_generative(model_output, target):
	return custom_mse_loss(model_output, target)

def calculate_loss_generative(model_output, target):
	predicted_sequence, mu, logvar = model_output
	kl_divergence = kl_loss(mu, logvar)
	mse_loss = custom_mse_loss(predicted_sequence, target)
	return kl_divergence + mse_loss

global_step_train = 0
global_step_valid = 0
global_fig_count = 0

def train(model, dataloader, optimizer, loss_fn, save_fig=False, writer=None, train_autoencoder=True, train_categorizer=True):
	global global_step_train
	global global_fig_count
	model = model.train()
	num_steps = len(dataloader)

	loss_history = np.empty(num_steps)
	loss_history_cel = np.empty(num_steps)
	acc_history = np.empty(num_steps)
	loss_cel = nn.CrossEntropyLoss()

	prediction_loss = None
	if type(model) is Autoencoder:
		prediction_loss = calculate_loss_non_generative
	elif type(model) is VariationalAutoencoder:
		prediction_loss = calculate_loss_generative

	epsilon = None
	if torch.cuda.is_available():
		epsilon = torch.tensor(1e-7).cuda()
	else:
		epsilon = torch.tensor(1e-7)

	for i_batch, sample_batched in enumerate(dataloader):
		print("{}/{}".format(i_batch + 1, num_steps), end="\r")

		input_tensor, reversed_input, teacher_input, output = sample_batched["input"], sample_batched["reversed_input"], sample_batched["teacher_input"], sample_batched["label"]

		model_output = model(input=input_tensor, teacher_input=teacher_input)
		loss = prediction_loss(model_output, reversed_input)

		loss_history[i_batch] = loss.item()

		if writer:
			writer.add_scalar("raw/loss/reconstruction", global_step=global_step_train, scalar_value=loss.item())

			if save_fig:
				fig = plt.figure(0)
				one_input = reversed_input[0].detach().cpu().numpy()
				one_input = one_input.reshape(one_input.shape[0])

				one_output = predicted_sequence[0].detach().cpu().numpy()
				one_output = one_output.reshape(one_output.shape[0])
				
				plt.plot(one_input, label="expected")
				plt.plot(one_output, label="predicted")
				plt.legend()
				writer.add_figure("raw/fig/reconstruction", figure=fig, global_step=global_fig_count)
				global_fig_count += 1

		if train_autoencoder:
			optimizer[0].zero_grad()
			loss.backward()
			optimizer[0].step()

		del loss # Necessary?

		model.freeze_autoencoder()
	
		predicted_category = model(input=input_tensor, teacher_input=None, use_predictor=True)
		loss = loss_cel(input=predicted_category+epsilon, target=output)
	
		loss_history_cel[i_batch] = loss.item()
		acc = calc_acc(input=predicted_category, target=output)
		acc_history[i_batch] = acc

		if writer:
			writer.add_scalar("raw/loss/categorization", global_step=global_step_train, scalar_value=loss.item())
			writer.add_scalar("raw/acc/categorization", global_step=global_step_train, scalar_value=acc)

		if train_categorizer:
			optimizer[1].zero_grad()
			loss.backward()
			optimizer[1].step()

		del loss # Necessary?

		model.unfreeze_autoencoder()

		global_step_train += 1

	if writer:
		writer.add_scalar("per_epoch/loss/categorization", global_step=global_step_train // num_steps, scalar_value=loss_history_cel.mean())
		writer.add_scalar("per_epoch/loss/reconstruction", global_step=global_step_train // num_steps, scalar_value=loss_history.mean())
		writer.add_scalar("per_epoch/acc/categorization", global_step=global_step_train // num_steps, scalar_value=acc_history.mean())
			
	return {"loss_reconstruction": loss_history.mean(), "loss_categorization": loss_history_cel.mean(), "accuracy_categorization": acc_history.mean()}

def evaluate(epoch, model, dataloader, loss_fn, start_of_sequence=-1, writer=None):
	global global_step_valid
	model.eval()
	
	num_steps = len(dataloader)

	loss_history = np.empty(num_steps)
	loss_history_cel = np.empty(num_steps)
	acc_history = np.empty(num_steps)
	loss_cel = nn.CrossEntropyLoss()

	prediction_loss = None
	if type(model) is Autoencoder:
		prediction_loss = calculate_loss_non_generative
	elif type(model) is VariationalAutoencoder:
		prediction_loss = calculate_loss_generative

	with torch.no_grad():
		for i_batch, sample_batched in enumerate(dataloader):
			print("{}/{}".format(i_batch + 1, num_steps), end="\r")
			input_tensor, reversed_input, _, output = sample_batched["input"], sample_batched["reversed_input"], sample_batched["teacher_input"], sample_batched["label"]

			predicted_category = model(input=input_tensor, teacher_input=None, use_predictor=True)
			loss = loss_cel(input=predicted_category, target=output)

			loss_history_cel[i_batch] = loss.item()
			acc = calc_acc(input=predicted_category, target=output)
			acc_history[i_batch] = acc

			if writer:
				writer.add_scalar("val_raw/acc/categorization", global_step=global_step_valid, scalar_value=acc)

			del loss # Necessary?

			global_step_valid += 1

		if writer:
			writer.add_scalar("val_per_epoch/loss/categorization", global_step=global_step_valid // num_steps, scalar_value=loss_history_cel.mean())
			writer.add_scalar("val_per_epoch/acc/categorization", global_step=global_step_valid // num_steps, scalar_value=acc_history.mean())

	return {"loss_categorization": loss_history_cel.mean(), "accuracy_categorization": acc_history.mean()}

def get_dict_string(d, prefix=""):
	result = prefix
	for k, v in d.items():
		result += " {}: {:.4f},".format(k, v)

	return result[:-1]

def get_comment_string(args):
	comment = "gen_" if args.use_variational_autoencoder else "non_gen_"
	comment += "b{}_".format(args.batch_size)
	comment += "db{}_".format(args.top_db)
	comment += "hs{}_".format(args.hidden_size)
	comment += "fc_hd{}_".format(args.fc_hidden_dim)
	comment += "lstm_" if args.use_lstm else "gru_"
	comment += "s{}_".format(args.shrink)
	comment += "e{}_".format(args.epochs)
	comment += "c{}".format(args.coins)
	return comment

def main(args):
	if not args.seed:
		args.seed = int(time.time()) 

	print("Random seed: {}".format(args.seed))
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

	if args.cudnn_deterministic:
		print("Running in CuDNN deterministic mode")
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	print("Worker thread count: {}".format(max(0, args.cpu_count)))

	torch.set_num_threads(max(0, args.cpu_count))
	cuda_available = torch.cuda.is_available()

	writer = SummaryWriter(comment=get_comment_string(args))
	model_save_dir_name = writer.log_dir.split("/")[-1]
	model_save_path = None

	writer.add_scalar("constants/seed", scalar_value=args.seed)

	if args.save:
		model_save_path = os.path.join(args.save, model_save_dir_name)
		os.makedirs(model_save_path)

	complete_dataset = NewCoinDataset(args)
	num_examples = len(complete_dataset)
	validation_dataset_size = int(args.val_split * num_examples)

	training_dataset, validation_dataset = torch.utils.data.random_split(complete_dataset, [num_examples - validation_dataset_size, validation_dataset_size])
	
	print("Training dataset length: {}".format(len(training_dataset)))
	print("Validation dataset length: {}".format(len(validation_dataset)))
	# print("Test dataset length: {}".format(len(test_dataset)))

	training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True, collate_fn=Collator())	
	validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True, num_workers=0)
	# test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0, drop_last=(test_batch_size > 1))

	model = None
	if args.use_variational_autoencoder:
		model = VariationalAutoencoder(hidden_dim=args.hidden_size, feature_dim=1, num_coins=complete_dataset.get_num_loaded_coins(), args=args)
	else:
		model = Autoencoder(hidden_dim=args.hidden_size, feature_dim=1, num_coins=complete_dataset.get_num_loaded_coins(), args=args)

	print("Using: {}".format(type(model).__name__))

	if cuda_available:
		print("Moving model to gpu...")
		model = model.cuda()

	opti = [optim.Adam(model.get_autoencoder_param(), lr=0.005), optim.Adam(model.get_predictor_param(), lr=0.01)]
	schedulers = [optim.lr_scheduler.MultiStepLR(opti[0], milestones=np.arange(args.epochs)[::20]), optim.lr_scheduler.MultiStepLR(opti[1], milestones=np.append([5, 10, 15, 20], np.arange(20, args.epochs)[::30]))]
	loss_fn = custom_mse_loss

	num_epochs = args.epochs

	print(model)
	print("Num parameters: {}".format(model.num_parameters()))

	num_epochs_no_improvements = 0
	best_val_loss = np.inf
	no_improvements_patience = 5
	no_improvements_min_epochs = 10

	if args.mode == "train":
		for current_epoch in range(num_epochs):

			start_time = time.time()
			train_history = train(model=model, dataloader=training_dataloader, optimizer=opti, loss_fn=custom_mse_loss, save_fig=args.save_figures ,writer=writer)
			end_time = time.time()

			print("Elapsed training time: {:.2f} seconds".format(end_time - start_time))
			
			start_time = time.time()
			validation_history = evaluate(epoch=current_epoch+1, model=model, dataloader=validation_dataloader, loss_fn=custom_mse_loss, writer=writer)
			end_time = time.time()

			print("Elapsed validation time: {:.2f} seconds".format(end_time - start_time))

			print("Epoch {}/{}:".format(current_epoch + 1, num_epochs))
			print(get_dict_string(train_history, "train: "))
			print(get_dict_string(validation_history, "val: "))
			print("---")

			# schedulers[0].step(train_history["loss_mean"])
			# schedulers[1].step(validation_history["accuracy"])

			if args.save:
				torch.save(model.state_dict(), os.path.join(model_save_path, "{:04d}.weights".format(current_epoch + 1)))

	if args.mode == "tsne":
		from sklearn.manifold import TSNE
		assert (args.weights), "No weights file specified!"

		device = torch.device('cpu')
		model.load_state_dict(torch.load(args.weights, map_location=device))
		model.eval()
		# Use training examples for now to test
		num_examples_per_class = complete_dataset.get_num_coins_per_class()
		coins = args.coins
		colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'][:len(args.coins)]

		plot_colors = []
		[plot_colors.extend(x*num_examples_per_class) for x in colors]

		fig = plt.figure()
		ax = fig.add_subplot(111)#, projection="3d")

		i = 0
		all_encodings = torch.empty(num_examples_per_class*len(coins), args.hidden_size)
		with torch.no_grad():
			for coin in coins:
				coin_data = complete_dataset.get_data_for_coin_type(coin=coin, num_examples=num_examples_per_class)

				for data in coin_data:
					print("\rCoin {}: {}/{}".format(coin, (i % num_examples_per_class) + 1, num_examples_per_class), end="")
					input_tensor = data["input"]
					encoded_input = model(input=input_tensor.view(1, input_tensor.shape[0], input_tensor.shape[1]), return_hidden=True)

					all_encodings[i] = encoded_input[0]
					i += 1
				print("")
			embedded = TSNE(n_components=2).fit_transform(all_encodings.numpy())

		ax.scatter(embedded[:,0], embedded[:,1], c=plot_colors, alpha=0.5)
		plt.show()

	if args.mode == "confusion":
		from sklearn.metrics import confusion_matrix
		assert (args.weights), "No weights file specified!"

		device = torch.device('cpu')
		model.load_state_dict(torch.load(args.weights, map_location=device))
		model.eval()

		num_examples_per_class = complete_dataset.get_num_coins_per_class()
		coins = args.coins

		expected = []
		predicted = []

		with torch.no_grad():
			for coin in coins:
				coin_data = complete_dataset.get_data_for_coin_type(coin=coin, num_examples=num_examples_per_class)

				for i, data in enumerate(coin_data):
					print("\rCoin {}: {}/{}".format(coin, i + 1, num_examples_per_class), end="")

					input_tensor, label = data["input"], data["label"]
					expected.append(coins[label])

					predicted_category = model(input=input_tensor.view(1, input_tensor.shape[0], input_tensor.shape[1]), use_predictor=True, teacher_input=None)
					predicted_category = predicted_category.cpu().numpy()

					predicted.append(coins[np.argmax(predicted_category)])
				print("")

		confusion_matrix = confusion_matrix(expected, predicted, labels=coins)
		print(confusion_matrix)
		print(np.divide(confusion_matrix, num_examples_per_class))

	if args.mode == "infer":
		raise Exception("This mode needs to be reimplemented") # Remove if reimplemented
		
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

	required_arguments = parser.add_argument_group('required arguments')

	required_arguments.add_argument("-p", "--path", type=str, required=True, help="Path to hdf5 data file.")

	parser.add_argument("--use_variational_autoencoder", action="store_true", help="Uses a variational autoencoder model")
	parser.add_argument("-m", "--mode", type=str, default="train", help="Mode of the script. Can be either 'train', 'tsne', 'confusion' or 'infer'. Default 'train'")
	parser.add_argument("-c", "--cpu_count", type=int, default=0, help="Number of worker threads to use. Default 0")
	parser.add_argument("-b", "--batch_size", type=int, default=128, help="Batch size. Default 1")
	parser.add_argument("-lstm", "--use_lstm", type=bool, default=True, help="Use lstm or gru. Default True = use lstm")
	parser.add_argument("--val_split", type=float, default=0.1, help="Validation split. Default is 0.1")
	parser.add_argument("-s", "--shrink", type=int, default=16, help="Shrinking factor. Selects data every s steps from input.")
	parser.add_argument("-hs", "--hidden_size", type=int, default=64, help="Size of LSTM/GRU hidden layer.")
	parser.add_argument("-fc_hd", "--fc_hidden_dim", type=int, default=100, help="Hidden dimension size of predictor fully connected layer. Default 100")
	parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs")
	parser.add_argument("--save", type=str, default=None, help="Specify save folder for weight files. Default: None")
	parser.add_argument("-w", "--weights", type=str, default=None, help="Model weights file. Only used for 'tsne' mode. Default: None")
	parser.add_argument("--top_db", type=int, default=5, help="Only used if --rosa is specified. Value under which audio is considered as silence at beginning/end.")
	parser.add_argument("--coins", nargs="+", default=[1, 2, 5, 20, 50, 100, 200], help="Use only specified coin types. Possible values: 1, 2, 5, 20, 50, 100, 200. Default uses all coins.")
	parser.add_argument("--num_examples", type=int, default=None, help="Number of used coin data examples from each class for training. Default uses the minimum number of all used classes.")
	parser.add_argument("--save_figures", action="store_true", help="Save figures of reconstructed time series.")
	parser.add_argument("--seed", type=int, default=None, help="Initializes Python, Numpy and Torch with this random seed. !!NOTE: Before running the script export PYTHONHASHSEED=0 as environment variable.!!")
	parser.add_argument("--cudnn_deterministic", action="store_true", help="Sets CuDNN into deterministic mode. This might impact perfromance.")

	args = parser.parse_args()

	main(args)
