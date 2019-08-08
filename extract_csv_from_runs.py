import os
import webbrowser
import argparse

def extract_seeds_from_dir(path):
	dirs = os.listdir(path)
	dirs = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), dirs))

	seeds = []
	for folder in dirs:
		seed_str = folder.split("_")[-1][4:]
		seeds.append(seed_str)

	return seeds

def open_seeds_from_base(base, seeds):
	for seed in seeds:
		url = base.format(seed)
		webbrowser.open(url, new=2)

def main(args):
	seeds = extract_seeds_from_dir(args.seed_dir)
	open_seeds_from_base(args.base, seeds)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--seed_dir", type=str)
	parser.add_argument("--base", type=str)

	args = parser.parse_args()
	main(args)
