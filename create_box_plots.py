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
		#csv_files = list(filter(lambda x: x.endswith(".csv"), csv_files))

		values = []
		for csv_file in csv_files:
			data = pandas.read_csv(os.path.join(csv_path, csv_file))
			idx_of_max = data["Value"].idxmax()
			values.append(float(data.iloc[idx_of_max]["Value"]) * 100)

		method_values.append(values)

	method_labels = []
	for method in methods:
		if method == "cnn":
			method_labels.append("CNN 1024 s1")
		if method == "cnn_4096":
			method_labels.append("CNN 4096 s1")
		if method == "cnn_4096_s4":
			method_labels.append("CNN 4096 s4")
		elif method == "enc_dec":
			method_labels.append("Enc-Dec")
		elif method == "simple_rnn":
			method_labels.append("Simple RNN")
		elif method == "enc_dec_windowed":
			method_labels.append("Enc-Dec 1024 s1")
		elif method == "enc_dec_windowed_4096":
			method_labels.append("Enc-Dec 4096 s1")
		elif method == "enc_dec_windowed_4096_s4":	
			method_labels.append("Enc-Dec 4096 s2")
		elif method == "simple_rnn_windowed":
			method_labels.append("Simple RNN 1024 s1")
		elif method == "simple_rnn_windowed_4096":
			method_labels.append("Simple RNN 4096 s1")
		elif method == "simple_rnn_windowed_4096_s4":
			method_labels.append("Simple RNN 4096 s4")

	fig, ax = plt.subplots()
	ax.boxplot(method_values, labels=method_labels)
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			 rotation_mode="anchor")
	plt.xlabel("Method")
	plt.ylabel("Best validation accuracy in %")
	plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("-p", "--path", type=str, required=True, help="Path to folder containing data for the different methods.")
	args = parser.parse_args()

	main(args)
