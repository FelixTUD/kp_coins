import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss

from utils.metrics import custom_mse_loss, categorization_acc
from utils.summarizer import summarize_values

from sessions.CoinSession import CoinSession
from model import CNNCategorizer

class CNN_Session(CoinSession):
	def __init__(self, args, training_dataloader, 
							 num_loaded_coins,
							 validation_dataloader=None, 
							 test_dataloader=None,
							 value_summarize_fn=summarize_values):
		super(CNN_Session, self).__init__(args, training_dataloader, 
												validation_dataloader=validation_dataloader, 
												test_dataloader=test_dataloader,
							 					value_summarize_fn=value_summarize_fn)

		self.model = CNNCategorizer(feature_dim=1, num_coins=num_loaded_coins, args=args)

		self.categorization_loss = CrossEntropyLoss()
		self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

		if torch.cuda.is_available():
			print("Moving model to gpu...")
			self.model.cuda()
			self.epsilon = torch.tensor(1e-7).cuda()
		else:
			self.epsilon = torch.tensor(1e-7)

		self.num_total_train_steps_per_epoch = len(self.training_dataloader) 

		if self.validation_dataloader:
			self.num_total_validation_steps_per_epoch = len(self.validation_dataloader) 

		if self.test_dataloader:
			self.num_total_test_steps_per_epoch = len(self.test_dataloader) 

	def comment_string(self):
		comment = "cnn_"
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

	def on_train_loop_start(self, current_epoch):
		self.model.train()
		print("-------------")
		print("Current epoch: {}".format(current_epoch + 1))

	def inner_loop(self, **kwargs):
		batch_num = kwargs["batch_num"]
		batch_content = kwargs["batch_content"]
		existing_results = kwargs["existing_results"]
		training_mode = kwargs["training"]

		print("{} batch: {}/{}".format("Training" if training_mode else "Evaluating", batch_num + 1, self.num_total_train_steps_per_epoch), end="\r")
		
		input_tensor, output = batch_content["input"], batch_content["label"]
		input_tensor = input_tensor.unsqueeze(1) # Introduce channel dimension, we have just 1 channel (=feature_dim)
		
		predicted_category = self.model(input=input_tensor)

		predicted_category = predicted_category.squeeze(1) # Remove channel dimension again
		loss = self.categorization_loss(input=predicted_category + self.epsilon, target=output)
		existing_results["{}/categorization_loss".format("train" if training_mode else "val")].append(loss.item())

		acc = categorization_acc(input=predicted_category, target=output)
		existing_results["{}/categorization_acc".format("train" if training_mode else "val")].append(acc)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		del loss # Necessary?

	def train_inner(self, **kwargs):
		kwargs["training"] = True
		self.inner_loop(**kwargs)

	def on_evaluate_loop_start(self, current_epoch):
		self.model.eval()

	def evaluate_inner(self, **kwargs):
		kwargs["training"] = False
		self.inner_loop(**kwargs)

	def on_evaluate_loop_finished(self, current_epoch):
		print("")
		print("Evaluation results:")

	def on_train_loop_finished(self, current_epoch):
		print("")
