#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=5:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1      # limit to one node
#SBATCH --cpus-per-task=4  # number of processor cores (i.e. threads)
#SBATCH --partition hpdlf
#SBATCH --mem-per-cpu=2500M   # memory per CPU core
#SBATCH -J "kp_training"   # job name
#SBATCH -A p_kpml

module load Python
source /scratch/p_kpml/venv/bin/activate

python /scratch/p_kpml/playground_marvin/torch_coin.py -b 10 -e 150 -p /scratch/p_kpml/playground_marvin/coin_data/data.hdf5 --top_db 5 --coins 1 100 -hs 64 -s 16 -c 4

exit 0