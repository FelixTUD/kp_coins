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

class DecoderLSTMPred(nn.Module):
	def __init__(self, hidden_dim, feature_dim, args):
		super(DecoderLSTMPred, self).__init__()
		self.is_decoder = True
		self.args = args

		self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)
		self.fc = nn.Linear(hidden_dim, feature_dim)
		self.time_dist_act = nn.Tanh() if args.rosa else nn.Sigmoid()

		fc_hidden_dim = hidden_dim * 2
		self.pred_fc_h = nn.Linear(hidden_dim, fc_hidden_dim)
		self.relu = nn.ReLU()
		self.pred_fc_h2 = nn.Linear(fc_hidden_dim, 2 if self.args.debug else 7)

		self.eval_mode = False

	def forward(self, input, initial):
		if self.eval_mode and self.is_decoder:
			seq_length = input.shape[1]

			output = torch.empty(input.shape).to(input.device)

			# Use detach() to not generate a huge flow graph and run out of memory
			for i in range(seq_length):
				partial_reconstruction, initial = self.lstm(input.data[:, i:i + 1, :], initial)
				partial_reconstruction = partial_reconstruction.detach()
				initial = (initial[0].detach(), initial[1].detach())

				output.data[:, i:i+1, :] = self.time_dist_act(self.fc(partial_reconstruction).detach()).detach().data
				input.data[:, i+1:i+2, :] = output.data[:, i:i+1, :]

			return output
		else:
			if self.is_decoder:
				reconstruction, _ = self.lstm(input, initial)
				unpacked, _ = nn.utils.rnn.pad_packed_sequence(reconstruction, batch_first=True) # [1] is tensor of original lengths
				predicted =  self.time_dist_act(self.fc(unpacked))

				return predicted
			else:
				return self.pred_fc_h2(self.relu(self.pred_fc_h(initial[0][0])))
				# return self.pred_fc_c(initial[1][0])

	def get_autoencoder_param(self):
		return list(self.lstm.parameters()) + list(self.fc.parameters())

	def get_predictor_param(self):
		return list(self.lstm.parameters()) + list(self.pred_fc_h.parameters()) + list(self.pred_fc_h2.parameters())

	def set_decoder_mode(self, toggle):
		self.is_decoder = toggle


	def set_eval_mode(self, toggle):
		self.eval_mode = toggle


class Autoencoder(nn.Module):
	def __init__(self, hidden_dim, feature_dim, args):
		super(Autoencoder, self).__init__()

		self.use_lstm = args.use_lstm
		self.hidden_dim = hidden_dim

		if self.use_lstm:
			self.encoder = EncoderLSTM(hidden_dim, feature_dim)
			#self.decoder = DecoderLSTM(hidden_dim, feature_dim, activation_function)
			self.decoder = DecoderLSTMPred(hidden_dim, feature_dim, args)
		else:
			self.encoder = EncoderGRU(hidden_dim, feature_dim)
			self.decoder = DecoderGRU(hidden_dim, feature_dim)

	def forward(self, input, teacher_input=None, return_hidden=False):
		_, last_hidden = self.encoder(input)
		if return_hidden:
			if self.use_lstm:
				result = torch.empty(2, self.hidden_dim)

				result[0] = last_hidden[0].detach().cpu()
				result[1] = last_hidden[1].detach().cpu()
				return result
			else:
				return last_hidden.detach().cpu()
		
		reconstructed = self.decoder(teacher_input, last_hidden)

		return reconstructed

	def get_autoencoder_param(self):
		return list(self.encoder.parameters()) + self.decoder.get_autoencoder_param()

	def get_predictor_param(self):
		return list(self.encoder.parameters()) + self.decoder.get_predictor_param()

	def num_parameters(self):
		return sum(p.numel() for p in self.parameters())

	def set_decoder_mode(self, toggle):
		self.decoder.set_decoder_mode(toggle)

	def set_eval_mode(self, toggle):
		self.decoder.set_eval_mode(toggle)
