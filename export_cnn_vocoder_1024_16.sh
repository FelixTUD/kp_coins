#!/bin/bash

export DATA_PATH=/warm_archive/ws/s0403431-kp_ml_data/1024_16.hdf5
export MODE=train
export NUM_EPOCHS=200
export BATCH_SIZE=96
export SAVE_PATH=vocoder_results/weights/cnn_1024_16
export TOP_DB=2
export HIDDEN_SIZE=100
export FC_HIDDEN_SIZE=100
export SHRINK=1
export WINDOW_SIZE=1024
export WINDOW_GAP=1024
export LEARNING_RATE=0.002
export EXTRA_ARGS="-a cnn --log_dir vocoder_results/stats/cnn_1024_16"
