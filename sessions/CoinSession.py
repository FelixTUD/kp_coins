import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from collections import defaultdict
import os

from utils.Session import Session

class CoinSession(Session):
	def __init__(self, args, training_dataloader,  
							 value_summarize_fn,
							 validation_dataloader=None, 
							 test_dataloader=None):
		super(CoinSession, self).__init__(epochs=args.epochs)

		self.args = args

		self._assert_type(training_dataloader, DataLoader)
		self.training_dataloader = training_dataloader

		if validation_dataloader:
			self._assert_type(validation_dataloader, DataLoader)
			self.validation_dataloader = validation_dataloader
		else:
			self.validation_dataloader = None

		if test_dataloader:
			self._assert_type(test_dataloader, DataLoader)
			self.test_dataloader = test_dataloader
		else:
			self.test_dataloader = None

		self.create_new_exisiting_results()
		self.value_summarize_fn = value_summarize_fn

		comment_string = self.comment_string()
		self.writer = SummaryWriter(log_dir=os.path.join(args.log_dir, self.args.extra_name + comment_string), comment=self.args.extra_name + comment_string, flush_secs=30)
		
		if self.args.save:
			model_save_dir_name = self.writer.log_dir.split("/")[-1]
			self.model_save_path = os.path.join(self.args.save, model_save_dir_name)
			os.makedirs(self.model_save_path, exist_ok=True)

	def __del__(self):
		self.writer.close()

	def create_new_exisiting_results(self):
		self.existing_results = defaultdict(list)

	def comment_string(self):
		# Default comment. Should be overwritten by each child class.
		comment = "b{batch_size}_"
		comment += "db{top_db}_"
		comment += "s{shrink}_"
		comment += "e{epochs}_"
		comment += "c{coins}_"
		comment += "seed{seed}"
		return self.fill_comments_with_args(comment)

	def log(self, result, current_epoch):
		for key, value in result.items():
			print("{}: {:.4f}".format(key, value))
			self.writer.add_scalar(key, global_step=current_epoch, scalar_value=value)

	def _assert_type(self, check, type):
		assert(isinstance(check, type))

	def train_inner(self, **kwargs):
		raise NotImplementedError()

	def evaluate_inner(self, **kwargs):
		print("Warning: evaluate() called, but evaluate_inner() not implemented.")

	def test_inner(self, **kwargs):
		print("Warning: test() called, but test_inner() not implemented.")

	def on_train_loop_start(self):
		pass

	def on_train_loop_finished(self):
		pass
	
	def on_evaluate_loop_start(self):
		pass

	def on_evaluate_loop_finished(self):
		pass

	def on_epoch_finished(self, current_epoch):
		if self.args.save:
			if self.args.no_state_dict:
				torch.save(self.model, os.path.join(self.model_save_path, "{:04d}.model".format(current_epoch + 1)))
			else:
				torch.save(self.model.state_dict(), os.path.join(self.model_save_path, "{:04d}.weights".format(current_epoch + 1)))

	def train(self, current_epoch):
		self.create_new_exisiting_results()

		self.on_train_loop_start(current_epoch)

		for batch_num, batch_content in enumerate(self.training_dataloader):
			self.train_inner(batch_num=batch_num, batch_content=batch_content, existing_results=self.existing_results, current_epoch=current_epoch) 

		self.on_train_loop_finished(current_epoch)

		return self.summarize_results()

	def evaluate(self, current_epoch):
		self.create_new_exisiting_results()

		self.on_evaluate_loop_start(current_epoch)

		with torch.no_grad():
			for batch_num, batch_content in enumerate(self.validation_dataloader):
				self.evaluate_inner(batch_num=batch_num, batch_content=batch_content, existing_results=self.existing_results) 

		self.on_evaluate_loop_finished(current_epoch)

		return self.summarize_results()

	def summarize_results(self):
		summarized_results = {}
		for key, values in self.existing_results.items():
			summarized_values = self.value_summarize_fn(values)
			summarized_results[key] = summarized_values

		return summarized_results

	def fill_comments_with_args(self, comment):
		return comment.format(**vars(self.args))
