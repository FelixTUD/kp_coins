import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np
# import matplotlib.pyplot as plt
import pandas 
import sys
from dataset import CoinDatasetLoader, CoinDataset
from model import Autoencoder

def custom_mse_loss(y_pred, y_true):
	return ((y_true-y_pred)**2).sum(1).mean()

def main(mode="train"):
	torch.set_num_threads(4)
	cuda_available = torch.cuda.is_available()

	dataset_loader = CoinDatasetLoader(path_to_hdf5="coin_data/data.hdf5", validation_split=0.1, test_split=0.1)

	training_dataset = dataset_loader.get_dataset("training")
	validation_dataset = dataset_loader.get_dataset("validation")
	test_dataset = dataset_loader.get_dataset("test")

	print("Training dataset length: {}".format(len(training_dataset)))
	print("Validation dataset length: {}".format(len(validation_dataset)))
	print("Test dataset length: {}".format(len(test_dataset)))

	training_batch_size = 10
	validation_batch_size = 1
	test_batch_size = 1

	training_dataloader = DataLoader(training_dataset, batch_size=training_batch_size, shuffle=True, num_workers=0, drop_last=True)
	validation_dataloader = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=0, drop_last=(validation_batch_size > 1))
	test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0, drop_last=(test_batch_size > 1))

	training_dataset_length = int(len(training_dataset) / training_batch_size)
	validation_dataset_length = int(len(validation_dataset) / validation_batch_size)
	test_dataset_length = int(len(test_dataset) / test_batch_size)

	model = Autoencoder(hidden_dim=2**6, feature_dim=1, use_lstm=False, activation_function=nn.Tanhshrink())
	if cuda_available:
		print("Moving model to gpu...")
		model = model.cuda()

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

	if mode == "train":
		for current_epoch in range(num_epochs):
			training_loss_history[:] = 0
			validation_loss_history[:] = 0

			for i_batch, sample_batched in enumerate(training_dataloader):
				print("{}/{}".format(i_batch + 1, training_dataset_length), end="\r")

				input_tensor, output = sample_batched["input"], sample_batched["label"]

				start_of_sequence = torch.empty(training_batch_size, 1, 1)
				start_of_sequence[:,:,0] = -1
				if cuda_available:
					start_of_sequence = start_of_sequence.cuda()

				teacher_input = torch.flip(torch.cat((start_of_sequence, input_tensor[:,:-1,:]), 1), dims=(1, 2))

				predicted_sequence = model(input=input_tensor, teacher_input=teacher_input)

				loss = loss_fn(predicted_sequence, output)

				training_loss_history[i_batch] = loss.item()

				opti.zero_grad()

				loss.backward()

				opti.step()

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

			print("Epoch {}/{}: loss: {:5f}".format(current_epoch + 1, num_epochs, training_loss_history.mean()))

			# print("Epoch {}/{}: loss: {:5f}, val_loss: {:5f}".format(current_epoch + 1, num_epochs, training_loss_history.mean(), val_loss))

			# if num_epochs_no_improvements == no_improvements_patience and no_improvements_min_epochs < current_epoch:
			# 	print("No imprevements in val loss for 3 epochs. Aborting training.")
			# 	break

	if mode == "infer":
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
	main(sys.argv[1])