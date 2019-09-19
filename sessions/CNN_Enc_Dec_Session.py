import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import os

from utils.metrics import custom_mse_loss, categorization_acc
from utils.summarizer import summarize_values

from sessions.CoinSession import CoinSession
from model import CNNAutoencoder, JustCNNAutoencoder, CNNPredictor


class CNN_Enc_Dec_Session(CoinSession):
	def __init__(self, args, training_dataloader, 
							 num_loaded_coins,
							 validation_dataloader=None, 
							 test_dataloader=None,
							 value_summarize_fn=summarize_values):
		super(CNN_Enc_Dec_Session, self).__init__(args, training_dataloader, 
													validation_dataloader=validation_dataloader, 
													test_dataloader=test_dataloader,
							 						value_summarize_fn=value_summarize_fn)

		if args.freeze:
			self.model = JustCNNAutoencoder(feature_dim=1, num_coins=num_loaded_coins, args=args)
			self.predictor = CNNPredictor(feature_dim=1, num_coins=num_loaded_coins, args=args)
		else:
			self.model = CNNAutoencoder(feature_dim=1, num_coins=num_loaded_coins, args=args)

		self.reconstruction_loss = custom_mse_loss
		self.categorization_loss = CrossEntropyLoss()

		if args.freeze:
			self.optimizers = [optim.Adam(self.model.get_encoder_param() + self.model.get_decoder_param(), lr=self.args.learning_rate), 
							   optim.Adam(self.predictor.parameters(), lr=self.args.learning_rate)]
		else:
			self.optimizers = [optim.Adam(self.model.get_encoder_param() + self.model.get_decoder_param(), lr=self.args.learning_rate), 
							   optim.Adam(self.model.get_encoder_param() + self.model.get_predictor_param(), lr=self.args.learning_rate)]

		print("Parameter Count: {}".format(self.model.num_parameters()))

		if torch.cuda.is_available():
			print("Moving model to gpu...")
			self.model.cuda()
			if args.freeze:
				self.predictor.cuda()
			self.epsilon = torch.tensor(1e-7).cuda()
		else:
			self.epsilon = torch.tensor(1e-7)

		self.num_total_train_steps_per_epoch = len(self.training_dataloader) 

		if self.validation_dataloader:
			self.num_total_validation_steps_per_epoch = len(self.validation_dataloader) 

		if self.test_dataloader:
			self.num_total_test_steps_per_epoch = len(self.test_dataloader) 

	def comment_string(self):
		comment = "cnn_enc_dec_"
		if self.args.freeze:
			comment = "freeze_"
		comment += "b{batch_size}_"
		comment += "lr{learning_rate}_"
		comment += "db{top_db}_"
		comment += "ws{window_size}_"
		comment += "wg{window_gap}_"
		comment += "s{shrink}_"
		comment += "e{epochs}_"
		comment += "c{coins}_"
		comment += "seed{seed}"
		comment = self.fill_comments_with_args(comment)

		return comment

	def on_epoch_finished(self, current_epoch):
		if self.args.save:
			if self.args.no_state_dict:
				if self.args.freeze:
					torch.save(self.model, os.path.join(self.model_save_path, "{:04d}.autoencoder.model".format(current_epoch + 1)))
					torch.save(self.predictor, os.path.join(self.model_save_path, "{:04d}.predictor.model".format(current_epoch + 1)))
				else:
					torch.save(self.model, os.path.join(self.model_save_path, "{:04d}.model".format(current_epoch + 1)))
			else:
				if self.args.freeze:
					torch.save(self.model.state_dict(), os.path.join(self.model_save_path, "{:04d}.autoencoder.weights".format(current_epoch + 1)))
					torch.save(self.predictor.state_dict(), os.path.join(self.model_save_path, "{:04d}.predictor.weights".format(current_epoch + 1)))
				else:
					torch.save(self.model.state_dict(), os.path.join(self.model_save_path, "{:04d}.weights".format(current_epoch + 1)))

	def on_train_loop_start(self, current_epoch):
		self.model.train()
		print("-------------")
		print("Current epoch: {}".format(current_epoch + 1))

	def train_inner(self, **kwargs):
		batch_num = kwargs["batch_num"]
		batch_content = kwargs["batch_content"]
		existing_results = kwargs["existing_results"]

		print("Training batch: {}/{}".format(batch_num + 1, self.num_total_train_steps_per_epoch), end="\r")

		input_tensor, reversed_input, output = batch_content["input"], batch_content["reversed_input"], batch_content["label"]
		input_tensor = input_tensor.unsqueeze(1)

		if self.args.architecture_split: 
			if current_epoch < self.args.architecture_split:
				model_output = self.model(input=input_tensor)
		
				loss = self.reconstruction_loss(model_output, input_tensor)

				existing_results["train/reconstruction_loss"].append(loss.item())

				self.optimizers[0].zero_grad()
				loss.backward()
				self.optimizers[0].step()

				del loss # Necessary?
				return
		else:
			model_output = self.model(input=input_tensor)
		
			loss = self.reconstruction_loss(model_output, input_tensor)

			existing_results["train/reconstruction_loss"].append(loss.item())

			self.optimizers[0].zero_grad()
			loss.backward()
			self.optimizers[0].step()

			del loss # Necessary?
	
		if self.args.freeze:
			encoded = self.model(input=input_tensor, use_predictor=True).detach()
			predicted_category = self.predictor(encoded)
		else:
			predicted_category = self.model(input=input_tensor, use_predictor=True)
		loss = self.categorization_loss(input=predicted_category + self.epsilon, target=output)
	
		existing_results["train/categorization_loss"].append(loss.item())
		acc = categorization_acc(input=predicted_category, target=output)
		existing_results["train/categorization_acc"].append(acc)

		self.optimizers[1].zero_grad()
		loss.backward()
		self.optimizers[1].step()

		del loss # Necessary?

	def on_evaluate_loop_start(self, current_epoch):
		self.model.eval()

	def evaluate_inner(self, **kwargs):
		batch_num = kwargs["batch_num"]
		batch_content = kwargs["batch_content"]
		existing_results = kwargs["existing_results"]

		print("Evaluating batch: {}/{}".format(batch_num + 1, self.num_total_validation_steps_per_epoch), end="\r")

		input_tensor, output = batch_content["input"], batch_content["label"]
		input_tensor = input_tensor.unsqueeze(1)

		if self.args.freeze:
			encoded = self.model(input=input_tensor, use_predictor=True).detach()
			predicted_category = self.predictor(encoded)
		else:
			predicted_category = self.model(input=input_tensor, use_predictor=True)
		loss = self.categorization_loss(input=predicted_category + self.epsilon, target=output)
	
		existing_results["val/categorization_loss"].append(loss.item())
		acc = categorization_acc(input=predicted_category, target=output)
		existing_results["val/categorization_acc"].append(acc)

	def on_evaluate_loop_finished(self, current_epoch):
		print("")
		print("Evaluation results:")

	def on_train_loop_finished(self, current_epoch):
		print("")
