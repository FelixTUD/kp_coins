#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=24:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1      # limit to one node
#SBATCH --cpus-per-task=2  # number of processor cores (i.e. threads)
#SBATCH --partition hpdlf
#SBATCH --mem-per-cpu=8000M   # memory per CPU core
#SBATCH -J "kp_training"   # job name
#SBATCH -A p_kpml

module load Python
source /scratch/p_kpml/venv/bin/activate

tensorboard --logdir=/scratch/p_kpml/playground_marvin/runs --port=6666 > /dev/null &
python /scratch/p_kpml/playground_marvin/torch_coin.py --no_state_dict -fc_hd $FC_HIDDEN_SIZE --save $SAVE_PATH -b $BATCH_SIZE -e $NUM_EPOCHS -p /scratch/p_kpml/playground_marvin/coin_data/data.hdf5 --val_split 0.2 --top_db $TOP_DB -hs $HIDDEN_SIZE -s 16 -c 4 -m $MODE -ws $WINDOW_SIZE $EXTRA_ARGS

exit 0
