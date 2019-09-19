#!/bin/bash

export MODE=train
export NUM_EPOCHS=200
export BATCH_SIZE=96
export SAVE_PATH=results/weights/enc_dec_windowed_4096_s4
export TOP_DB=2
export HIDDEN_SIZE=100
export FC_HIDDEN_SIZE=100
export SHRINK=4
export WINDOW_SIZE=4096
export WINDOW_GAP=4096
export LEARNING_RATE=0.003
export EXTRA_ARGS="-a enc_dec --log_dir results/stats/enc_dec_windowed_4096_s4 --use_windows"
