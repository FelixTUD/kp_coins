#!/bin/bash

export MODE=train
export NUM_EPOCHS=200
export BATCH_SIZE=24
export SAVE_PATH=results/weights/enc_dec_s2
export TOP_DB=2
export HIDDEN_SIZE=100
export FC_HIDDEN_SIZE=100
export SHRINK=2
export WINDOW_SIZE=1024
export WINDOW_GAP=1024
export LEARNING_RATE=0.002
export EXTRA_ARGS="-a enc_dec --log_dir results/stats/enc_dec_s2"
