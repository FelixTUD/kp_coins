import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt
import pandas 

class ToTensor(object):
	def __init__(self, use_cuda=False):
		super(ToTensor, self).__init__()

		self.use_cuda = use_cuda

	def __call__(self, sample):
		input, output = sample['input'], sample['output']

		input = torch.from_numpy(input).float()
		output = torch.from_numpy(output).float()
        
		if self.use_cuda:
			input = input.cuda()
			output = output.cuda()

		return {'input': input, 'output': output}

class FlightDataset(Dataset):
	def __init__(self, csv_file, input_window_size, output_window_size, transform=None, split="train"):
		super(FlightDataset, self).__init__()

		self.transform = transform
		self.input_window_size = input_window_size
		self.output_window_size = output_window_size
		self.window_size = self.input_window_size + self.output_window_size

		passenger_csv = pandas.read_csv(csv_file, header=0)
		passengers = passenger_csv["Passengers"].values
		# passengers = self.normalize(passengers)

		if split == "train":
			self.passenger_data, _ = np.split(passengers, [0, int(passengers.size * 0.8)])[1:]
		elif split == "validation":
			_, self.passenger_data = np.split(passengers, [0, int(passengers.size * 0.8) - (self.input_window_size + self.output_window_size)])[1:]

		# print("Loaded {} data points".format(self.passenger_data.size))

		num_it = self.passenger_data.size - self.window_size

		self.data = []

		for idx in range(num_it):
			window_data = self.passenger_data[idx:idx + self.window_size]
			self.data.append(self.normalize(window_data).reshape(self.window_size, 1))

		self.length = len(self.data)

	def __len__(self):
		return self.length

	def normalize(self, timeseries):
		return (timeseries - np.min(timeseries)) / (np.max(timeseries) - np.min(timeseries))

	def __getitem__(self, idx):
		window_data = self.data[idx]
		training_input = window_data[:self.input_window_size]
		training_output = window_data[self.input_window_size:]

		sample = {"input" : training_input, "output" : training_output}

		if self.transform:
			sample = self.transform(sample)

		return sample

class EncoderGRU(nn.Module):
	def __init__(self, hidden_dim, feature_dim):
		super(EncoderGRU, self).__init__()

		self.gru = nn.GRU(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)

	def forward(self, input):
		return self.gru(input)

class DecoderGRU(nn.Module):
	def __init__(self, hidden_dim, feature_dim):
		super(DecoderGRU, self).__init__()

		self.gru = nn.GRU(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)
		self.fc = nn.Linear(hidden_dim, feature_dim)

		self.activation = nn.Sigmoid()
		# self.activation = nn.ReLU()

	def forward(self, input, initial):
		reconstruction, _ = self.gru(input, initial)
		return self.activation(self.fc(reconstruction))

class EncoderLSTM(nn.Module):
	def __init__(self, hidden_dim, feature_dim):
		super(EncoderLSTM, self).__init__()

		self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)

	def forward(self, input):
		return self.lstm(input)

class DecoderLSTM(nn.Module):
	def __init__(self, hidden_dim, feature_dim):
		super(DecoderLSTM, self).__init__()

		self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)
		self.fc = nn.Linear(hidden_dim, feature_dim)

		self.activation = nn.Sigmoid()
		# self.activation = nn.ReLU()

	def forward(self, input, initial):
		reconstruction, _ = self.lstm(input, initial)
		return self.activation(self.fc(reconstruction))

class Autoencoder(nn.Module):
	def __init__(self, hidden_dim, feature_dim, use_lstm=True):
		super(Autoencoder, self).__init__()

		if use_lstm:
			self.encoder = EncoderLSTM(hidden_dim, feature_dim)
			self.decoder = DecoderLSTM(hidden_dim, feature_dim)
		else:
			self.encoder = EncoderGRU(hidden_dim, feature_dim)
			self.decoder = DecoderGRU(hidden_dim, feature_dim)

	def forward(self, input, teacher_input):
		encoded_input, last_hidden = self.encoder(input)
		reconstructed = self.decoder(teacher_input, last_hidden)

		return reconstructed

	def num_parameters(self):
		return sum(p.numel() for p in self.parameters())

def custom_mse_loss(y_pred, y_true):
	return ((y_true-y_pred)**2).sum(1).mean()

torch.set_num_threads(4)

cuda_available = torch.cuda.is_available()

num_to_learn = 10
num_to_predict = 5

transform = ToTensor(use_cuda=cuda_available)
training_dataset = FlightDataset("airline-passengers.csv", input_window_size=num_to_learn, output_window_size=num_to_predict, transform=transform, split="train")
validation_dataset = FlightDataset("airline-passengers.csv", input_window_size=num_to_learn, output_window_size=num_to_predict, transform=transform, split="validation")

print("Training dataset length: {}".format(len(training_dataset)))
print("Validation dataset length: {}".format(len(validation_dataset)))

training_batch_size = 5
assert ((len(training_dataset) % training_batch_size) == 0)
validation_batch_size = 1
assert ((len(validation_dataset) % validation_batch_size) == 0)

training_dataloader = DataLoader(training_dataset, batch_size=training_batch_size, shuffle=True, num_workers=3)
validation_dataloader = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=3)

training_dataset_length = int(len(training_dataset) / training_batch_size)
validation_dataset_length = int(len(validation_dataset) / validation_batch_size)

model = Autoencoder(hidden_dim=2**6, feature_dim=1, use_lstm=False)
if cuda_available:
	print("Moving model to gpu...")
	model = model.cuda()

opti = optim.Adam(model.parameters())
mse_loss = custom_mse_loss

num_epochs = 100

# num_parameters = sum(p.numel() for p in model.parameters())
print(model)
print("Num parameters: {}".format(model.num_parameters()))

training_loss_history = np.empty(training_dataset_length)
validation_loss_history = np.empty(validation_dataset_length)

num_epochs_no_improvements = 0
best_val_loss = np.inf
no_improvements_patience = 5
no_improvements_min_epochs = 10

for current_epoch in range(num_epochs):
	training_loss_history[:] = 0
	validation_loss_history[:] = 0

	for i_batch, sample_batched in enumerate(training_dataloader):

		input_tensor, output = sample_batched["input"], sample_batched["output"]
		teacher_input = torch.cat((input_tensor[:,-1,:].reshape(training_batch_size, 1, 1), output[:,:-1,:]), 1)

		predicted_sequence = model(input=input_tensor, teacher_input=teacher_input)

		loss = mse_loss(predicted_sequence, output)

		training_loss_history[i_batch] = loss.item()

		opti.zero_grad()

		loss.backward()

		opti.step()

	for val_i_batch, val_sample_batch in enumerate(validation_dataloader):

		input_tensor, output = val_sample_batch["input"], val_sample_batch["output"]
		teacher_input = input_tensor[:,-1,:].reshape(validation_batch_size, 1, 1)

		# Predict iteratively

		for _ in range(num_to_predict):

			partial_predicted_sequence = model(input=input_tensor, teacher_input=teacher_input)

			teacher_input = torch.cat((teacher_input, partial_predicted_sequence[:,-1,:].reshape(validation_batch_size, 1, 1)), 1)

		loss = mse_loss(partial_predicted_sequence, output)

		validation_loss_history[val_i_batch] = loss.item()

	val_loss = validation_loss_history.mean()

	if best_val_loss < val_loss:
		num_epochs_no_improvements += 1
	else:
		num_epochs_no_improvements = 0
		torch.save(model.state_dict(), "rae_teacher_forcing_weights.pt")

	best_val_loss = np.minimum(best_val_loss, val_loss)

	print("Epoch {}/{}: loss: {:5f}, val_loss: {:5f}".format(current_epoch + 1, num_epochs, training_loss_history.mean(), val_loss))

	if num_epochs_no_improvements == no_improvements_patience and no_improvements_min_epochs < current_epoch:
		print("No imprevements in val loss for 3 epochs. Aborting training.")
		break

model.load_state_dict(torch.load("rae_teacher_forcing_weights.pt"))

gt_val = []
pred_val = []

try:
	for i, val_sample_batch in enumerate(validation_dataset):

		if i % num_to_predict == 0:

			input_tensor, output = val_sample_batch["input"], val_sample_batch["output"]
			input_tensor = input_tensor.reshape(1, input_tensor.shape[0], input_tensor.shape[1])
			teacher_input = input_tensor[:,-1,:].reshape(1, 1, 1)

			gt_val += (list(output.numpy().reshape(num_to_predict)))

			# Predict iteratively

			for _ in range(num_to_predict):

				partial_predicted_sequence = model(input=input_tensor, teacher_input=teacher_input)

				teacher_input = torch.cat((teacher_input, partial_predicted_sequence[:,-1,:].reshape(1, 1, 1)), 1)

			pred_val += (list(partial_predicted_sequence[0].cpu().detach().numpy().reshape(num_to_predict)))
except:
	pass

plt.plot(np.arange(len(gt_val)), gt_val, label="gt")
plt.plot(np.arange(len(pred_val)), pred_val, label="pred")
plt.legend()
plt.savefig("val_pred_plot.pdf", format="pdf")
plt.show()