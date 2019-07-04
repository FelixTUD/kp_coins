import torch

def custom_mse_loss(y_pred, y_true):
	return ((y_true-y_pred)**2).sum(1).mean()

def categorization_acc(input, target):
	return (torch.argmax(input, 1) == target).sum().item() / input.shape[0]

def kl_loss(mu, logvar):
	return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def calculate_loss_generative(model_output, target):
	predicted_sequence, mu, logvar = model_output
	kl_divergence = kl_loss(mu, logvar)
	mse_loss = custom_mse_loss(predicted_sequence, target)
	return kl_divergence + mse_loss