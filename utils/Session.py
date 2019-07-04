
class Session(object):
	def __init__(self, epochs, args=None):
		self.epochs = epochs
		self.args = args

	def startup(self):
		print("Warning: startup() called, but startup() not implemented.")

	def cleanup(self):
		print("Warning: cleanup() called, but cleanup() not implemented.")

	def train(self):
		raise NotImplementedError()

	def test(self):
		print("Warning: test() called, but test() not implemented.")

	def evaluate(self):
		print("Warning: evaluate() called, but evalute() not implemented.")

	def log(self, result, current_epoch=None):
		print("Warning: log called, but log() not implemented.")

	def run(self, evaluate=True, test=True):
		self.startup()

		for current_epoch in range(self.epochs):
			train_result = self.train()
			self.log(train_result, current_epoch)

			if evaluate:
				evaluate_result = self.evaluate()
				self.log(evaluate_result, current_epoch)

		if test:
			test_result = self.test()
			self.log(test_result)		

		self.cleanup()