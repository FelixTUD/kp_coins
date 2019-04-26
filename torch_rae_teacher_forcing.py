import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

class Encoder(nn.Module):
	def __init__(self, hidden_dim, feature_dim):
		super(Encoder, self).__init__()

		self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)

	def forward(self, input):
		return self.lstm(input)

class Decoder(nn.Module):
	def __init__(self, hidden_dim, feature_dim):
		super(Decoder, self).__init__()

		self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)
		self.fc = nn.Linear(hidden_dim, feature_dim)

		self.relu = nn.ReLU()

	def forward(self, input, h_0, c_0):
		reconstruction, _ = self.lstm(input, (h_0, c_0))
		return self.relu(self.fc(reconstruction))

class Autoencoder(nn.Module):
	def __init__(self, hidden_dim, feature_dim):
		super(Autoencoder, self).__init__()

		self.encoder = Encoder(hidden_dim, feature_dim)
		self.decoder = Decoder(hidden_dim, feature_dim)

	def forward(self, input, teacher_input):
		encoded_input, (h_n, c_n) = self.encoder(input)
		reconstructed = self.decoder(teacher_input, h_0=h_n, c_0=c_n)

		return reconstructed

def new_training_example(batch_size):
	length = np.random.randint(10, 100)
	training_batch = torch.empty(batch_size, length, 1)

	for i in range(batch_size):
		start = np.random.randint(10, 50)
		training_example = torch.arange(start=start, end=start + length).float()
		# training_example = (training_example - torch.min(training_example) / (torch.max(training_example) - torch.min(training_example)))

		training_batch[i] = training_example.reshape(training_example.shape[0], 1)

	return training_batch


model = Autoencoder(hidden_dim=100, feature_dim=1)
opti = optim.Adam(model.parameters())
mse_loss = nn.MSELoss()

num_epochs = 10000
batch_size = 10

num_parameters = sum(p.numel() for p in model.parameters())
print(model)
print("Num parameters: {}".format(num_parameters))

loss_history = np.empty(num_epochs)

for current_epoch in range(num_epochs):
	
	training_batch = new_training_example(batch_size)
	reversed_batch = training_batch.flip(1)

	expected_output = torch.cat((training_batch, torch.zeros(10, 1, 1)), 1)
	teacher_input = torch.cat((torch.zeros(10, 1, 1), reversed_batch), 1)

	predicted_sequence = model(input=training_batch, teacher_input=teacher_input)

	loss = mse_loss(predicted_sequence, expected_output)

	loss_history[current_epoch] = loss.item()

	print("Epoch {}/{}: loss: {:5f}".format(current_epoch + 1, num_epochs, loss.item()))

	opti.zero_grad()

	loss.backward()

	opti.step()

torch.save(model.state_dict(), "rae_teacher_forcing_weights.pt")

plt.plot(np.arange(loss_history.size), loss_history)
plt.show()


