#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=24:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=1      # limit to one node
#SBATCH --cpus-per-task=2  # number of processor cores (i.e. threads)
#SBATCH --partition haswell
#SBATCH --mem-per-cpu=8000M   # memory per CPU core
#SBATCH -J "conversion"   # job name
#SBATCH -A p_kpml

module load Python
source /scratch/p_kpml/venv/bin/activate

python convert_data_into_hdf5.py 1024_16.hdf5 /scratch/p_kpml/vocoder_muenzen/1024_16/
python convert_data_into_hdf5.py 1024_2.hdf5 /scratch/p_kpml/vocoder_muenzen/1024_2/
python convert_data_into_hdf5.py 1024_32.hdf5 /scratch/p_kpml/vocoder_muenzen/1024_32/
python convert_data_into_hdf5.py 1024_4.hdf5 /scratch/p_kpml/vocoder_muenzen/1024_4/
python convert_data_into_hdf5.py 1024_8.hdf5 /scratch/p_kpml/vocoder_muenzen/1024_8/
python convert_data_into_hdf5.py 256_16.hdf5 /scratch/p_kpml/vocoder_muenzen/256_16/
python convert_data_into_hdf5.py 256_2.hdf5 /scratch/p_kpml/vocoder_muenzen/256_2/
python convert_data_into_hdf5.py 256_32.hdf5 /scratch/p_kpml/vocoder_muenzen/256_32/
python convert_data_into_hdf5.py 256_4.hdf5 /scratch/p_kpml/vocoder_muenzen/256_4/
python convert_data_into_hdf5.py 256_8.hdf5 /scratch/p_kpml/vocoder_muenzen/256_8/

exit 0
