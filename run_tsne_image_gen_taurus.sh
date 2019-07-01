#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=10:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1      # limit to one node
#SBATCH --cpus-per-task=2  # number of processor cores (i.e. threads)
#SBATCH --partition hpdlf
#SBATCH --mem-per-cpu=8000M   # memory per CPU core
#SBATCH -J "tsne_gen"   # job name
#SBATCH -A p_kpml

module load Python
source /scratch/p_kpml/venv/bin/activate

sh run_tsne_image_gen.sh

exit 0
