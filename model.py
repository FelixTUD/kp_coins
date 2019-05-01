import torch
import torch.nn as nn

class EncoderGRU(nn.Module):
	def __init__(self, hidden_dim, feature_dim):
		super(EncoderGRU, self).__init__()

		self.gru = nn.GRU(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)

	def forward(self, input):
		return self.gru(input)

class DecoderGRU(nn.Module):
	def __init__(self, hidden_dim, feature_dim, activation):
		super(DecoderGRU, self).__init__()

		self.gru = nn.GRU(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)
		self.fc = nn.Linear(hidden_dim, feature_dim)

		self.activation = activation

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
	def __init__(self, hidden_dim, feature_dim, activation):
		super(DecoderLSTM, self).__init__()

		self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)
		self.fc = nn.Linear(hidden_dim, feature_dim)

		self.activation = activation

		self.eval_mode = False

	def forward(self, input, initial):
		if self.eval_mode:
			seq_length = input.shape[1]

			output = torch.empty(input.shape).to(input.device)

			# Use detach() to not generate a huge flow graph and run out of memory
			for i in range(seq_length):
				partial_reconstruction, initial = self.lstm(input.data[:, i:i + 1, :], initial)
				partial_reconstruction = partial_reconstruction.detach()
				initial = (initial[0].detach(), initial[1].detach())

				if self.activation:
					output.data[:, i:i+1, :] = self.activation(self.fc(partial_reconstruction).detach()).detach().data
				else:
					output.data[:, i:i+1, :] = self.fc(partial_reconstruction).detach().data
				input.data[:, i+1:i+2, :] = output.data[:, i:i+1, :]

			return output
		else:
			reconstruction, _ = self.lstm(input, initial)
			if self.activation:
				return self.activation(self.fc(reconstruction))
			else:
				return self.fc(reconstruction)

	def set_eval_mode(self, toggle):
		self.eval_mode = toggle

class Autoencoder(nn.Module):
	def __init__(self, hidden_dim, feature_dim, activation_function, use_lstm=True):
		super(Autoencoder, self).__init__()

		if use_lstm:
			self.encoder = EncoderLSTM(hidden_dim, feature_dim)
			self.decoder = DecoderLSTM(hidden_dim, feature_dim, activation_function)
		else:
			self.encoder = EncoderGRU(hidden_dim, feature_dim)
			self.decoder = DecoderGRU(hidden_dim, feature_dim, activation_function)

	def forward(self, input, teacher_input):
		encoded_input, last_hidden = self.encoder(input)
		reconstructed = self.decoder(teacher_input, last_hidden)

		return reconstructed

	def num_parameters(self):
		return sum(p.numel() for p in self.parameters())

	def set_eval_mode(self, toggle):
		self.decoder.set_eval_mode(toggle)