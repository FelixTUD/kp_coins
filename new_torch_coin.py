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
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict

from sessions.Enc_Dec_Session import Enc_Dec_Session
from sessions.CNN_Session import CNN_Session
from sessions.Simple_RNN_Session import Simple_RNN_Session

from datasets.WindowedDataset import WindowedDataset
from datasets.UnWindowedDataset import UnWindowedDataset, Collator

def plot_confusion_matrix(cm, classes,
						  title=None,
						  cmap=plt.cm.Blues,
						  args=None):
	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
		   yticks=np.arange(cm.shape[0]),
		   # ... and label them with the respective list entries
		   xticklabels=classes, yticklabels=classes,
		   title=title,
		   ylabel='True label',
		   xlabel='Predicted label')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			 rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.2f'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
				ha="center", va="center",
				color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	
	if args and args.plot_title:
		plt.title(args.plot_title)

	if args and args.save:
		plt.savefig(os.path.join(args.save, "confusion.png"), format="png")
		plt.clf()
	else:
		plt.show()

def main(args):
	if not args.seed:
		args.seed = (int(time.time()*1000000) % (2**32)) 

	print("Random seed: {}".format(args.seed))
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

	if args.cudnn_deterministic:
		print("Running in CuDNN deterministic mode")
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	print("Number of additional worker threads: {}".format(max(0, args.cpu_count)))
	torch.set_num_threads(max(0, args.cpu_count))

	cuda_available = torch.cuda.is_available() and not args.run_cpu

	complete_dataset = None
	if args.architecture == "cnn" or args.use_windows:
		complete_dataset = WindowedDataset(args)
	else:
		complete_dataset = UnWindowedDataset(args)

	num_examples = len(complete_dataset)
	validation_dataset_size = int(args.val_split * num_examples)
	training_dataset, validation_dataset = torch.utils.data.random_split(complete_dataset, [num_examples - validation_dataset_size, validation_dataset_size])
	
	print("Training dataset length: {}".format(len(training_dataset)))
	print("Validation dataset length: {}".format(len(validation_dataset)))
	# print("Test dataset length: {}".format(len(test_dataset)))

	if args.architecture == "cnn":
		training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
		validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
	else:
		training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=Collator())	
		validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=Collator())

	# test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0, drop_last=(test_batch_size > 1))
	
	if args.mode == "train":

		session = None
		if args.architecture == "enc_dec":
			session = Enc_Dec_Session(args=args, 
									  training_dataloader=training_dataloader, 
									  validation_dataloader=validation_dataloader,
									  num_loaded_coins=complete_dataset.get_num_loaded_coins())

		elif args.architecture == "cnn":
			session = CNN_Session(args=args, 
								  training_dataloader=training_dataloader, 
								  validation_dataloader=validation_dataloader,
								  num_loaded_coins=complete_dataset.get_num_loaded_coins())

		elif args.architecture == "simple_rnn":
			session = Simple_RNN_Session(args=args, 
								  		 training_dataloader=training_dataloader, 
								  		 validation_dataloader=validation_dataloader,
								  		 num_loaded_coins=complete_dataset.get_num_loaded_coins())

		session.run(evaluate=True, test=False)

	if args.mode == "metrics":
		if args.save:
			os.makedirs(args.save, exist_ok=True)

		# Create tsne and confusion matrix and save at path from --save argument

		from sklearn.manifold import TSNE
		assert (args.weights), "No weights file specified!"

		if cuda_available:
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')
		model = None
		if args.no_state_dict:
			model = torch.load(args.weights, map_location=device)
		else:
			assert(args.no_state_dict), "State dict is necessary for metrics. Use --no_state_dict during training!"

		model.eval()

		coins = args.coins

		if args.architecture != "cnn":
			num_examples_per_class = complete_dataset.get_num_coins_per_class()
			
			colors = ['aqua', 'darkorange', 'cornflowerblue', 'darkblue', 'black', 'green', 'red'][:len(args.coins)]
			plot_colors = []

			print("Generating TSNE plot")

			final_encodings = None
			if args.tsne_set == "validation":
				encodings = defaultdict(list)
				validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True, num_workers=0)
				with torch.no_grad():
					for batch_content in validation_dataloader:
						input_tensor = batch_content["input"]
						coin_class = batch_content["label"] # Numeric from 0...numCoins
						coin_class = coins[coin_class]

						encoded_input = model(input=input_tensor, return_hidden=True)
						encodings[coin_class].append(encoded_input[0].numpy())

				flattened_encodings = []
				for i, coin in enumerate(coins):
					flattened_encodings.extend(encodings[coin])
					color = colors[i]
					for _ in range(len(encodings[coin])):
						plot_colors.append(color)

				flattened_encodings = np.array(flattened_encodings)
				print(flattened_encodings.shape)
				final_encodings = flattened_encodings

			elif args.tsne_set == "all":
				i = 0
				all_encodings = None #torch.empty(num_examples_per_class*len(coins), args.hidden_size)
				with torch.no_grad():
					for coin_num, coin in enumerate(coins):
						coin_data = complete_dataset.get_data_for_coin_type(coin=coin, num_examples=num_examples_per_class)

						for data in coin_data:
							print("\rCoin {}: {}/{}".format(coin, (i % num_examples_per_class) + 1, num_examples_per_class), end="")
							input_tensor = data["input"]
							if args.use_windows:
								encoded_input = model(input=input_tensor, return_hidden=True)
							else:
								encoded_input = model(input=input_tensor.view(1, input_tensor.shape[0], input_tensor.shape[1]), return_hidden=True)

							if all_encodings is None:
								all_encodings = torch.empty(num_examples_per_class*len(coins), encoded_input.shape[1])

							all_encodings[i] = encoded_input[0]
							plot_colors.append(colors[coin_num])
							i += 1
						print("")
				
				final_encodings = all_encodings.numpy()
			
			for perplexity in range(10, 140, 10):
				fig = plt.figure(figsize=(16, 9), dpi=120)
				ax = fig.add_subplot(111)#, projection="3d")

				print("Embedding with preplexity={}...".format(perplexity))
				embedded = TSNE(n_components=2, perplexity=perplexity).fit_transform(final_encodings)
				ax.scatter(embedded[:,0], embedded[:,1], c=plot_colors, alpha=0.5)

				# Create custom legend
				labels = ["1 ct", "2 ct", "5 ct", "20 ct", "50 ct", "1 €", "2 €"]
				legend_items = []
				for i, _ in enumerate(coins):
					legend_items.append(Line2D([0], [0], marker='o', color="w", label=labels[i], markerfacecolor=colors[i], markersize=15))
		 
				box = ax.get_position()
				ax.set_position([box.x0, box.y0 + box.height * 0.1,
		                 		box.width, box.height * 0.9])

				ax.legend(handles=legend_items, loc='upper center', bbox_to_anchor=(0.5, -0.05),
		          		  fancybox=True, shadow=True, ncol=len(coins))

				if args.plot_title:
					plt.title(args.plot_title)

				if args.save:
					plt.savefig(os.path.join(args.save, "tsne_{}.png".format(perplexity)), format="png")
					plt.clf()
				else:
					plt.show()

		from sklearn.metrics import confusion_matrix

		print("Generating confusion matrix")

		expected = []
		predicted = []

		validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True, num_workers=0)
		with torch.no_grad():
			for batch_content in validation_dataloader:
				input_tensor = batch_content["input"]
				coin_class = batch_content["label"] # Numeric from 0...numCoins

				if args.architecture == "cnn":
					input_tensor = input_tensor.unsqueeze(1) # Introduce channel dimension, we have just 1 channel (=feature_dim)
					predicted_category = model(input=input_tensor, use_predictor=True, teacher_input=None)
					predicted_category = predicted_category.squeeze(1) # Remove channel dimension again
				else:
					predicted_category = model(input=input_tensor, use_predictor=True, teacher_input=None)

				predicted_category = predicted_category.cpu().numpy()

				expected.append(coins[coin_class])
				predicted.append(coins[np.argmax(predicted_category)])

		confusion_matrix = confusion_matrix(expected, predicted, labels=coins)
		print(confusion_matrix)
		norm_cm = confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis]
		print(norm_cm)

		plot_confusion_matrix(norm_cm, coins, "Normalized Confusion Matrix", args=args)

	if args.mode == "roc":
		from sklearn.metrics import roc_curve, auc
		from itertools import cycle

		assert (args.weights), "No weights file specified!"

		if cuda_available:
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')
		model = None
		if args.no_state_dict:
			model = torch.load(args.weights, map_location=device)
		else:
			if args.use_variational_autoencoder:
				model = VariationalAutoencoder(hidden_dim=args.hidden_size, feature_dim=1, num_coins=complete_dataset.get_num_loaded_coins(), args=args)
			else:
				model = Autoencoder(hidden_dim=args.hidden_size, feature_dim=1, num_coins=complete_dataset.get_num_loaded_coins(), args=args)

			print("Using: {}".format(type(model).__name__))

			model.load_state_dict(torch.load(args.weights, map_location=device))
		model.eval()

		num_examples_per_class = complete_dataset.get_num_coins_per_class()
		coins = args.coins

		expected = []
		predicted = []

		with torch.no_grad():
			for i, data in enumerate(complete_dataset):
				print("\rTesting: {}/{}".format(i + 1, len(complete_dataset)), end="")

				input_tensor, label = data["input"], data["label"]
				expected.append(label)

				predicted_category = model(input=input_tensor.view(1, input_tensor.shape[0], input_tensor.shape[1]), use_predictor=True, teacher_input=None)
				predicted_category = predicted_category.cpu().numpy()

				predicted.append(np.argmax(predicted_category))

		fpr = dict()
		tpr = dict()
		roc_auc = dict()

		for class_index in range(len(coins)):
			expected = np.array(expected)
			predicted = np.array(predicted)
			one_vs_all_expected = np.where(expected == class_index, 1, 0)
			one_vs_all_predicted = np.where(predicted == class_index, 1, 0)

			fpr[class_index], tpr[class_index], _ = roc_curve(one_vs_all_expected, one_vs_all_predicted)
			roc_auc[class_index] = auc(fpr[class_index], tpr[class_index])

		colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'darkblue', 'black', 'green', 'red'])
		lw = 2
		plt.figure()
		for i, color in zip(range(len(coins)), colors):
			plt.plot(fpr[i], tpr[i], color=color, lw=lw,
					 label='ROC curve of class {0} (area = {1:0.2f})'
					 ''.format(coins[i], roc_auc[i]))

		plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic example')
		plt.legend(loc="lower right")
		plt.show()


if __name__ == "__main__":
	torch.multiprocessing.set_start_method("spawn")

	parser = argparse.ArgumentParser()

	required_arguments = parser.add_argument_group('required arguments')

	required_arguments.add_argument("-p", "--path", type=str, required=True, help="Path to hdf5 data file.")

	parser.add_argument("--use_variational_autoencoder", action="store_true", help="Uses a variational autoencoder model")
	parser.add_argument("-m", "--mode", type=str, choices=["train", "metrics"], default="train", help="Mode of the script. Can be either 'train' or 'metrics'. Default 'train'")
	parser.add_argument("-a", "--architecture", type=str, choices=["enc_dec", "cnn", "simple_rnn"], default="enc_dec", help="NN architecture to use. Default: enc_dec")
	parser.add_argument("--tsne_set", type=str, choices=["validation", "all"], default="all", help="What dataset to use for TSNE plot creation. All means training set + validation set. Default: all.")

	parser.add_argument("-c", "--cpu_count", type=int, default=0, help="Number of worker threads to use. Default 0")
	parser.add_argument("-b", "--batch_size", type=int, default=96, help="Batch size. Default 96")
	parser.add_argument("-lstm", "--use_lstm", type=bool, default=True, help="Use lstm or gru. Default True = use lstm")
	parser.add_argument("--val_split", type=float, default=0.1, help="Validation split. Default is 0.1")
	parser.add_argument("-s", "--shrink", type=int, default=16, help="Shrinking factor. Selects data every s steps from input. Default: 16")
	parser.add_argument("-hs", "--hidden_size", type=int, default=64, help="Size of LSTM/GRU hidden layer. Default: 64")
	parser.add_argument("-fc_hd", "--fc_hidden_dim", type=int, default=100, help="Hidden dimension size of predictor fully connected layer. Default 100")
	parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs")
	parser.add_argument("--save", type=str, default=None, help="Specify save folder for weight files or plots. Default: None")
	parser.add_argument("-w", "--weights", type=str, default=None, help="Model weights file. Only used for 'metrics' mode. Default: None")
	parser.add_argument("--top_db", type=int, default=2, help="Only used if --rosa is specified. Value under which audio is considered as silence at beginning/end. Default: 2")
	parser.add_argument("--coins", nargs="+", default=[1, 2, 5, 20, 50, 100, 200], help="Use only specified coin types. Possible values: 1, 2, 5, 20, 50, 100, 200. Default uses all coins.")
	parser.add_argument("--num_examples", type=int, default=None, help="Number of used coin data examples from each class for training. Default uses the minimum number of all used classes.")
	parser.add_argument("--save_figures", action="store_true", help="Save figures of reconstructed time series.")
	parser.add_argument("--seed", type=int, default=None, help="Initializes Python, Numpy and Torch with this random seed. !!NOTE: Before running the script export PYTHONHASHSEED=0 as environment variable.!!")
	parser.add_argument("--cudnn_deterministic", action="store_true", help="Sets CuDNN into deterministic mode. This might impact perfromance.")
	parser.add_argument("--no_state_dict", action="store_true", help="If set, saves the whole model instead of just the weights.")
	parser.add_argument("--run_cpu", action="store_true", help="If set, calculates on the CPU, even if GPU is available.") # not functional
	parser.add_argument("-ws", "--window_size", type=int, default=1024, help="Window size for training. Used if --use_windows is specified. Default 1024.")
	parser.add_argument("-wg", "--window_gap", type=int, default=1024, help="Gap between two consecutive windows. Default 1024.")
	parser.add_argument("--use_windows", action="store_true", help="If set, training uses a sliding window with window size specified by -ws. Default off, if using cnn defaults to on.")
	parser.add_argument("--save_plot", type=str, default=None, help="Save file name for plots from 'tsne' and 'confusion' modes. Default None")
	parser.add_argument("--plot_title", type=str, default=None, help="Title for 'tsne' and 'confusion' plots. Default None")
	parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate. Default 0.001")
	parser.add_argument("--log_dir", type=str, default="./runs/", help="Log directory for tensorboard data. Default ./runs/")
	parser.add_argument("--extra_name", type=str, default="", help="Extra string for tensorboard comment. Default None")

	args = parser.parse_args()

	main(args)
