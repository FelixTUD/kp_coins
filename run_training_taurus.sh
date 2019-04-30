#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=05:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1      # limit to one node
#SBATCH --cpus-per-task=1  # number of processor cores (i.e. threads)
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=5000M   # memory per CPU core
#SBATCH --partition hpdlf
#SBATCH -J "kp_training"   # job name
#SBATCH --mail-user=marvin.arnold@mailbox.tu-dresden.de   # email address
#SBATCH --mail-type=END,FAIL
#SBATCH -A p_ml_finanzen

module load Python
source /scratch/p_ml_finanzen/venv/bin/activate

python /scratch/p_ml_finanzen/kp_coin/torch_coin.py -p /scratch/p_ml_finanzen/kp_coin -m train -c 0 -b 25

exit 0