import os
import argparse

import pandas
import matplotlib.pyplot as plt

def main(args):
	methods = os.listdir(args.path)
	methods = list(filter(lambda x: os.path.isdir(os.path.join(args.path, x)), methods))

	method_values = []
	for method in methods:
		# Load all csv data files, extract the highest val acc
		csv_path = os.path.join(args.path, method, "box_plot_data")
		if not os.path.exists(csv_path):
			print("WARNING: Method {} does not have a box_plot_data/ folder.".format(method))
			continue

		csv_files = os.listdir(csv_path)
		csv_files = list(filter(lambda x: x.endswith(".csv"), csv_files))

		values = []
		for csv_file in csv_files:
			data = pandas.read_csv(os.path.join(csv_path, csv_file))
			idx_of_max = data["Value"].idxmax()
			values.append(float(data.iloc[idx_of_max]["Value"]) * 100)

		method_values.append(values)

	method_labels = []
	for method in methods:
		if method == "cnn":
			method_labels.append("CNN")
		elif method == "enc_dec":
			method_labels.append("Encoder-Decoder")
		elif method == "simple_rnn":
			method_labels.append("Simple RNN")

	plt.boxplot(method_values, labels=method_labels)
	plt.xlabel("Method")
	plt.ylabel("Best validation accuracy in %")
	plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("-p", "--path", type=str, required=True, help="Path to folder containing data for the different methods.")
	args = parser.parse_args()

	main(args)