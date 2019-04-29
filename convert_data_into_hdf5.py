import h5py
import os
import numpy as np

targets = [1, 2, 5, 20, 50, 100, 200]

save_file = h5py.File("data.hdf5")

all_coin_csv_files = os.listdir("coin_data")
all_coin_csv_files = list(filter(lambda x: x.endswith(".csv"), all_coin_csv_files))

def load_values_from_csv(file):
	return np.genfromtxt(file)

for coin_value in targets:
	print(coin_value)
	coin_value_group = save_file.create_group(str(coin_value))

	# g16
	g8_group = coin_value_group.create_group(str("g16"))
	print("g16")
	if coin_value == 200:
		all_coin_value_csv_files = list(filter(lambda x: x.startswith(str(coin_value) + "_tisch_g8_"), all_coin_csv_files))
		all_coin_value_csv_files = list(filter(lambda x: not x.startswith(str(coin_value) + "_tisch_g8_0"), all_coin_value_csv_files))
	else:
		all_coin_value_csv_files = list(filter(lambda x: x.startswith(str(coin_value) + "_tisch_g16"), all_coin_csv_files))

	for file in all_coin_value_csv_files:
		number = file.split(".")[0].split("_")[-1]
		number_group = g8_group.create_group(number)
		number_group.create_dataset("values", data=load_values_from_csv("coin_data/{}".format(file)), compression="gzip")

	# g8
	g8_group = coin_value_group.create_group(str("g8"))
	print("g8")
	if coin_value == 200:
		all_coin_value_csv_files = list(filter(lambda x: x.startswith(str(coin_value) + "_tisch_g8_0"), all_coin_csv_files))
	else:
		all_coin_value_csv_files = list(filter(lambda x: x.startswith(str(coin_value) + "_tisch_g8"), all_coin_csv_files))
	for file in all_coin_value_csv_files:
		number = file.split(".")[0].split("_")[-1]
		number_group = g8_group.create_group(number)
		number_group.create_dataset("values", data=load_values_from_csv("coin_data/{}".format(file)), compression="gzip")

