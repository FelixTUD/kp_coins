
class Session(object):
	def __init__(self, epochs, args=None):
		self.epochs = epochs
		self.args = args

	def startup(self):
		pass

	def cleanup(self):
		pass

	def train(self, current_epoch):
		raise NotImplementedError()

	def test(self):
		print("Warning: test() called, but test() not implemented.")

	def evaluate(self, current_epoch):
		print("Warning: evaluate() called, but evalute() not implemented.")

	def log(self, result, current_epoch=None):
		print("Warning: log called, but log() not implemented.")

	def on_epoch_start(self, current_epoch):
		pass

	def on_epoch_finished(self, current_epoch):
		pass

	def run(self, evaluate=True, test=True):
		self.startup()

		for current_epoch in range(self.epochs):
			self.on_epoch_start(current_epoch)

			train_result = self.train(current_epoch)
			if train_result:
				self.log(train_result, current_epoch)

			if evaluate:
				evaluate_result = self.evaluate(current_epoch)
				if evaluate_result:
					self.log(evaluate_result, current_epoch)

			self.on_epoch_finished(current_epoch)

		if test:
			test_result = self.test()
			if test_result:
				self.log(test_result)		

		self.cleanup()