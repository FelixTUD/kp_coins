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

		## Initialize weights

		nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
		nn.init.xavier_uniform_(self.lstm.weight_hh_l0)

	def forward(self, input):
		return self.lstm(input)

class Predictor(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(Predictor, self).__init__()

		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.bn_1 = nn.BatchNorm1d(hidden_dim)

		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.bn_2 = nn.BatchNorm1d(hidden_dim)

		#self.fc3 = nn.Linear(hidden_dim, hidden_dim)
		#self.bn_3 = nn.BatchNorm1d(hidden_dim)
		
		self.fc_out = nn.Linear(hidden_dim, output_dim)
		self.bn_out = nn.BatchNorm1d(output_dim)

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid() # To force outputs between 0-1 for logsoftmax. Might help with unexpected dips??

	def forward(self, input):
		input = self.relu(self.fc1(input))
		input = self.bn_1(input)

		input = self.relu(self.fc2(input))
		input = self.bn_2(input)

		#input = self.relu(self.fc3(input))
		#input = self.bn_3(input)

		# Old way
		# return self.sigmoid(self.fc_out(input))

		# New way
		input = self.sigmoid(self.fc_out(input))
		input = self.bn_out(input)

		return input

# class Predictor(nn.Module):
# 	def __init__(self, input_dim, hidden_dim, output_dim):
# 		super(Predictor, self).__init__()

# 		self.fc1 = nn.Linear(input_dim, hidden_dim)
# 		# self.bn_1 = nn.BatchNorm1d(hidden_dim)

# 		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
# 		# self.bn_2 = nn.BatchNorm1d(hidden_dim)

# 		#self.fc3 = nn.Linear(hidden_dim, hidden_dim)
# 		#self.bn_3 = nn.BatchNorm1d(hidden_dim)
		
# 		self.fc_out = nn.Linear(hidden_dim, output_dim)
# 		# self.bn_out = nn.BatchNorm1d(output_dim)

# 		self.relu = nn.ReLU()
# 		self.sigmoid = nn.Sigmoid() # To force outputs between 0-1 for logsoftmax. Might help with unexpected dips??

# 	def forward(self, input):
# 		input = self.relu(self.fc1(input))

# 		input = self.relu(self.fc2(input))

# 		#input = self.relu(self.fc3(input))
# 		#input = self.bn_3(input)

# 		# Old way
# 		# return self.sigmoid(self.fc_out(input))

# 		# New way
# 		input = self.fc_out(input)

# 		return input


class DecoderLSTM(nn.Module):
	def __init__(self, hidden_dim, feature_dim):
		super(DecoderLSTM, self).__init__()

		self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)
		self.fc = nn.Linear(hidden_dim, feature_dim)
		self.time_dist_act = nn.Tanh()
		self.sigmoid = nn.Sigmoid()

		## Initialize weights

		nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
		nn.init.xavier_uniform_(self.lstm.weight_hh_l0)

	def forward(self, input, initial):
		reconstruction, _ = self.lstm(input, initial)
		if type(reconstruction) is nn.utils.rnn.PackedSequence:
			unpacked, _ = nn.utils.rnn.pad_packed_sequence(reconstruction, batch_first=True) # second return is tensor of original lengths
			predicted =  self.time_dist_act(self.fc(unpacked))

			return predicted
		else:
			return self.time_dist_act(self.fc(reconstruction))

class VariationalIntermediate(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(VariationalIntermediate, self).__init__()

		self.mufc = nn.Linear(input_dim, output_dim)
		self.logvarfc = nn.Linear(input_dim, output_dim)

	def forward(self, input):
		mu = self.mufc(input)
		logvar = self.logvarfc(input)

		return mu, logvar

class VariationalAutoencoder(nn.Module):
	def __init__(self, hidden_dim, feature_dim, num_coins, args):
		super(VariationalAutoencoder, self).__init__()

		self.use_lstm = args.use_lstm
		self.hidden_dim = hidden_dim

		self.intermediate = VariationalIntermediate(hidden_dim, hidden_dim//2)

		# self.mufc = nn.Linear(hidden_dim, hidden_dim//2)
		# self.logvarfc = nn.Linear(hidden_dim, hidden_dim//2)

		if self.use_lstm:
			self.encoder = EncoderLSTM(hidden_dim, feature_dim)
			self.decoder = DecoderLSTM(hidden_dim//2, feature_dim)
		else:
			self.encoder = EncoderGRU(hidden_dim, feature_dim)
			self.decoder = DecoderGRU(hidden_dim//2, feature_dim)

		self.predictor = Predictor(input_dim=hidden_dim//2, hidden_dim=args.fc_hidden_dim, output_dim=num_coins)

	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		
		return mu + eps*std

	def forward(self, input, teacher_input=None, use_predictor=False, return_hidden=False):
		_, last_hidden = self.encoder(input)

		# mu = self.mufc(last_hidden[0][0])
		# logvar = self.logvarfc(last_hidden[0][0])

		mu, logvar = self.intermediate(last_hidden[0][0])

		z = self.reparameterize(mu=mu, logvar=logvar)

		if return_hidden:
			return z.detach().cpu()

		if use_predictor:
			return self.predictor(z)
		else:
			batch_size, hidden_size = z.shape
			z = z.view(1, batch_size, hidden_size)

			# hidden_c = torch.randn_like(z)
			# nn.init.xavier_uniform_(hidden_c)

			hidden_in = (z, z)
			
			reconstructed = self.decoder(teacher_input, hidden_in)

			return reconstructed, mu, logvar

	def get_encoder_param(self):
		return list(self.encoder.parameters()) + list(self.intermediate.parameters())

	def get_decoder_param(self):
		return list(self.decoder.parameters())

	def get_predictor_param(self):
		return list(self.predictor.parameters())

	def num_parameters(self):
		return sum(p.numel() for p in self.parameters())

	def freeze_autoencoder(self):
		for param in self.encoder.parameters():
			param.requires_grad = False
		for param in self.decoder.parameters():
			param.requires_grad = False

	def unfreeze_autoencoder(self):
		for param in self.encoder.parameters():
			param.requires_grad = True
		for param in self.decoder.parameters():
			param.requires_grad = True

class SimpleRNN(nn.Module):
	def __init__(self, hidden_dim, feature_dim, num_coins, args):
		super(SimpleRNN, self).__init__()

		self.use_lstm = args.use_lstm
		self.hidden_dim = hidden_dim

		if self.use_lstm:
			self.encoder = EncoderLSTM(hidden_dim, feature_dim)
		else:
			self.encoder = EncoderGRU(hidden_dim, feature_dim)

		self.predictor = Predictor(input_dim=hidden_dim, hidden_dim=args.fc_hidden_dim, output_dim=num_coins)

	def forward(self, input, return_hidden=False, **kwargs):
		_, last_hidden = self.encoder(input)
		if return_hidden:
			if self.use_lstm:
				result = torch.empty(2, self.hidden_dim)

				result[0] = last_hidden[0].cpu()
				result[1] = last_hidden[1].cpu()
				return result
			else:
				return last_hidden.cpu()

		return self.predictor(last_hidden[0][0])

class JustAutoencoder(nn.Module):
	def __init__(self, hidden_dim, feature_dim, num_coins, args):
		super(JustAutoencoder, self).__init__()

		self.use_lstm = args.use_lstm
		self.hidden_dim = hidden_dim

		if self.use_lstm:
			self.encoder = EncoderLSTM(hidden_dim, feature_dim)
			self.decoder = DecoderLSTM(hidden_dim, feature_dim)
		else:
			self.encoder = EncoderGRU(hidden_dim, feature_dim)
			self.decoder = DecoderGRU(hidden_dim, feature_dim)

	def forward(self, input, teacher_input=None, use_predictor=False, return_hidden=False):
		_, last_hidden = self.encoder(input)
		if return_hidden:
			if self.use_lstm:
				result = torch.empty(2, self.hidden_dim)

				result[0] = last_hidden[0].cpu()
				result[1] = last_hidden[1].cpu()
				return result
			else:
				return last_hidden.cpu()

		if use_predictor:
			return last_hidden[0][0]
		else:
			return self.decoder(teacher_input, last_hidden)

	def get_encoder_param(self):
		return list(self.encoder.parameters())

	def get_decoder_param(self):
		return list(self.decoder.parameters())

	def num_parameters(self):
		return sum(p.numel() for p in self.parameters())

class Autoencoder(nn.Module):
	def __init__(self, hidden_dim, feature_dim, num_coins, args):
		super(Autoencoder, self).__init__()

		self.encoder_requires_grad = True

		self.use_lstm = args.use_lstm
		self.hidden_dim = hidden_dim

		if self.use_lstm:
			self.encoder = EncoderLSTM(hidden_dim, feature_dim)
			self.decoder = DecoderLSTM(hidden_dim, feature_dim)
		else:
			self.encoder = EncoderGRU(hidden_dim, feature_dim)
			self.decoder = DecoderGRU(hidden_dim, feature_dim)

		self.predictor = Predictor(input_dim=hidden_dim, hidden_dim=args.fc_hidden_dim, output_dim=num_coins)

	def forward(self, input, teacher_input=None, use_predictor=False, return_hidden=False):
		_, last_hidden = self.encoder(input)
		if return_hidden:
			if self.use_lstm:
				result = torch.empty(2, self.hidden_dim)

				result[0] = last_hidden[0].cpu()
				result[1] = last_hidden[1].cpu()
				return result
			else:
				return last_hidden.cpu()

		if use_predictor:
			return self.predictor(last_hidden[0][0])
		else:
			return self.decoder(teacher_input, last_hidden)

	def get_encoder_param(self):
		return list(self.encoder.parameters())

	def get_decoder_param(self):
		return list(self.decoder.parameters())

	def get_predictor_param(self):
		# return list(self.encoder.parameters()) + list(self.predictor.parameters())
		return list(self.predictor.parameters())

	def num_parameters(self):
		return sum(p.numel() for p in self.parameters())

	def toogle_freeze_encoder(self):
		self.encoder_requires_grad = not self.encoder_requires_grad
		for param in self.encoder.parameters():
			param.requires_grad = self.encoder_requires_grad
	
class CNNEncoder(nn.Module):
	def __init__(self, feature_dim):
		super(CNNEncoder, self).__init__()

		self.conv1 = ConvLayerDown(feature_dim, 16, 3, padding=1)

		self.conv2 = ConvLayerDown(16, 16, 3, padding=1)

		self.conv3 = ConvLayerDown(16, 16, 3, padding=1)

		self.conv4 = ConvLayerDown(16, 16, 3, padding=1)

		self.conv5 = ConvLayerDown(16, 16, 3, padding=1)

		self.conv6 = ConvLayerDown(16, 16, 3, padding=1)

	def forward(self, input):
		input = self.conv1(input)

		input = self.conv2(input)

		input = self.conv3(input)

		input = self.conv4(input)

		input = self.conv5(input)

		input = self.conv6(input)
		return input

class CNNPredictor(nn.Module):
	def __init__(self, feature_dim, num_coins, args):
		super(CNNPredictor, self).__init__()

		self.fc_out = nn.Linear(((args.window_size // 64)) * 16, num_coins)
		self.bnorm_out = nn.BatchNorm1d(num_coins)
		self.sigmoid = nn.Sigmoid()

	def forward(self, input):
		input = input.view(input.size(0), -1)

		input = self.fc_out(input)
		input = self.bnorm_out(input)
		input = self.sigmoid(input)
		#input = self.bnorm_out(input)
		return input

class CNNDecoder(nn.Module):
	def __init__(self, feature_dim):
		super(CNNDecoder, self).__init__()

		self.conv6 = ConvLayerUp(16, 16, 3, padding=1)

		self.conv5 = ConvLayerUp(16, 16, 3, padding=1)

		self.conv4 = ConvLayerUp(16, 16, 3, padding=1)

		self.conv3 = ConvLayerUp(16, 16, 3, padding=1)

		self.conv2 = ConvLayerUp(16, 16, 3, padding=1)

		self.conv1 = ConvLayerUp(16, 16, 3, padding=1)

		self.conv0 = ConvLayer(16, feature_dim, 3, padding=1)

		self.sigmoid = nn.Sigmoid()

	def forward(self, input):
		input = self.conv6(input)

		input = self.conv5(input)

		input = self.conv4(input)

		input = self.conv3(input)

		input = self.conv2(input)

		input = self.conv1(input)

		input = self.sigmoid(self.conv0(input))
		return input

class JustCNNAutoencoder(nn.Module):
	def __init__(self, feature_dim, num_coins, args):
		super(JustCNNAutoencoder, self).__init__()

		self.encoder = CNNEncoder(feature_dim)
		self.decoder = CNNDecoder(feature_dim)

	def forward(self, input, use_predictor=False, return_hidden=False, **kwargs):
		last = self.encoder(input)

		if return_hidden:
			return last.view(input.size(0), -1).cpu()

		if use_predictor:
			return last
		else:
			return self.decoder(last)

	def get_encoder_param(self):
		return list(self.encoder.parameters())

	def get_decoder_param(self):
		return list(self.decoder.parameters())

	def num_parameters(self):
		return sum(p.numel() for p in self.parameters())

class CNNAutoencoder(nn.Module):
	def __init__(self, feature_dim, num_coins, args):
		super(CNNAutoencoder, self).__init__()

		self.encoder = CNNEncoder(feature_dim)
		self.decoder = CNNDecoder(feature_dim)

		self.predictor = CNNPredictor(feature_dim=feature_dim, num_coins=num_coins, args=args)

	def forward(self, input, use_predictor=False, return_hidden=False, **kwargs):
		last = self.encoder(input)

		if return_hidden:
			return last.view(input.size(0), -1).cpu()

		if use_predictor:
			return self.predictor(last)
		else:
			return self.decoder(last)

	def get_encoder_param(self):
		return list(self.encoder.parameters())

	def get_decoder_param(self):
		return list(self.decoder.parameters())

	def get_predictor_param(self):
		# return list(self.encoder.parameters()) + list(self.predictor.parameters())
		return list(self.predictor.parameters())

	def num_parameters(self):
		return sum(p.numel() for p in self.parameters())

class ConvLayerUp(nn.Module):
	def __init__(self, in_channels, out_channels, filter_size, padding):
		super(ConvLayerUp, self).__init__()
		self.conv = nn.Conv1d(in_channels, out_channels, filter_size, padding=padding)
		self.bnorm = nn.BatchNorm1d(out_channels)
		self.relu = nn.Sigmoid()
		self.upsample = nn.Upsample(scale_factor=2)


	def forward(self, input):
		input = self.conv(input)
		input = self.bnorm(input)
		input = self.relu(input)
		input = self.upsample(input)
		return input

class ConvLayer(nn.Module):
	def __init__(self, in_channels, out_channels, filter_size, padding):
		super(ConvLayer, self).__init__()
		self.conv = nn.Conv1d(in_channels, out_channels, filter_size, padding=padding)
		self.bnorm = nn.BatchNorm1d(out_channels)
		self.relu = nn.ReLU()

	def forward(self, input):
		input = self.conv(input)
		input = self.bnorm(input)
		input = self.relu(input)
		return input

class ConvLayerDown(nn.Module):
	def __init__(self, in_channels, out_channels, filter_size, padding):
		super(ConvLayerDown, self).__init__()
		self.conv = nn.Conv1d(in_channels, out_channels, filter_size, padding=padding)
		self.bnorm = nn.BatchNorm1d(out_channels)
		self.relu = nn.Sigmoid()
		self.pool = nn.MaxPool1d(2)

	def forward(self, input):
		input = self.conv(input)
		input = self.bnorm(input)
		input = self.relu(input)
		input = self.pool(input)
		return input

class CNNCategorizer(nn.Module):
	def __init__(self, feature_dim, num_coins, args):
		super(CNNCategorizer, self).__init__()
		self.conv1 = ConvLayerDown(feature_dim, 64, 3, padding=1)

		self.conv2 = ConvLayerDown(64, 64, 3, padding=1)

		self.conv3 = ConvLayerDown(64, 64, 3, padding=1)

		self.conv4 = ConvLayerDown(64, 64, 3, padding=1)

		self.conv5 = ConvLayerDown(64, 64, 3, padding=1)

		self.conv6 = ConvLayerDown(64, 64, 3, padding=1)

		#self.apool = nn.AvgPool1d(kernel_size=64, padding=32)

		self.fc_out = nn.Linear((args.window_size // 64) * 64, num_coins)
		self.bnorm_out = nn.BatchNorm1d(num_coins)
		self.sigmoid = nn.Sigmoid()
		

	def forward(self, input, return_hidden=False, **kwargs):
		input = self.conv1(input)

		input = self.conv2(input)

		input = self.conv3(input)

		input = self.conv4(input)

		input = self.conv5(input)

		input = self.conv6(input)

		#input = self.apool(input)

		input = input.view(input.size(0), -1)
		if return_hidden:
			return input.cpu()

		input = self.fc_out(input)
		input = self.bnorm_out(input)
		input = self.sigmoid(input)
		#input = self.bnorm_out(input)
		return input

	def num_parameters(self):
		return sum(p.numel() for p in self.parameters())
