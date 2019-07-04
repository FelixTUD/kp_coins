from statistics import mean
import numpy as np

def summarize_values(values):
	if isinstance(values, np.ndarray):
		return values.mean()
	if isinstance(values, list):
		return mean(values)
	if isinstance(values, int) or isinstance(values, float):
		return values
	
	print("Warning: No default summarize method for key {} of type {}.\nPlease provide a custom value_summarize_fn.".format(key, type(values)))
	return None