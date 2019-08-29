
# Requirements


```
absl-py==0.7.1
cycler==0.10.0
future==0.17.1
grpcio==1.20.1
h5py==2.9.0
kiwisolver==1.1.0
Markdown==3.1
matplotlib==3.0.3
numpy==1.16.3
pandas==0.24.2
Pillow==6.0.0
protobuf==3.7.1
pyparsing==2.4.0
python-dateutil==2.8.0
pytz==2019.1
scikit-learn==0.20.3
scipy==1.2.1
six==1.12.0
tb-nightly==1.14.0a20190511
torch==1.1.0
torchvision==0.2.2.post3
Werkzeug==0.15.3
librosa==0.6.3
```

# Usage
Please use the updated and new script ```new_torch_coin.py```.
The following command line arguments are available:
```
usage: new_torch_coin.py [-h] -p PATH [--use_variational_autoencoder]
                         [-m MODE] [-a {enc_dec,cnn,simple_rnn}]
                         [-c CPU_COUNT] [-b BATCH_SIZE] [-lstm USE_LSTM]
                         [--val_split VAL_SPLIT] [-s SHRINK] [-hs HIDDEN_SIZE]
                         [-fc_hd FC_HIDDEN_DIM] [-e EPOCHS] [--save SAVE]
                         [-w WEIGHTS] [--top_db TOP_DB]
                         [--coins COINS [COINS ...]]
                         [--num_examples NUM_EXAMPLES] [--save_figures]
                         [--seed SEED] [--cudnn_deterministic]
                         [--no_state_dict] [--run_cpu] [-ws WINDOW_SIZE]
                         [-wg WINDOW_GAP] [--use_windows]
                         [--save_plot SAVE_PLOT] [--plot_title PLOT_TITLE]
                         [-lr LEARNING_RATE]
required arguments:
  -p PATH, --path PATH  Path to hdf5 data file.

optional arguments:
  -h, --help            show this help message and exit
  --use_variational_autoencoder
                        Uses a variational autoencoder model
  -m MODE, --mode MODE  Mode of the script. Can be either 'train', 'tsne',
                        'confusion' or 'infer'. Default 'train'
  -a {enc_dec,cnn,simple_rnn}, --architecture {enc_dec,cnn,simple_rnn}
                        NN architecture to use. Default: enc_dec
  -c CPU_COUNT, --cpu_count CPU_COUNT
                        Number of worker threads to use. Default 0
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size. Default 96
  -lstm USE_LSTM, --use_lstm USE_LSTM
                        Use lstm or gru. Default True = use lstm
  --val_split VAL_SPLIT
                        Validation split. Default is 0.1
  -s SHRINK, --shrink SHRINK
                        Shrinking factor. Selects data every s steps from
                        input. Default: 16
  -hs HIDDEN_SIZE, --hidden_size HIDDEN_SIZE
                        Size of LSTM/GRU hidden layer. Default: 64
  -fc_hd FC_HIDDEN_DIM, --fc_hidden_dim FC_HIDDEN_DIM
                        Hidden dimension size of predictor fully connected
                        layer. Default 100
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs
  --save SAVE           Specify save folder for weight files. Default: None
  -w WEIGHTS, --weights WEIGHTS
                        Model weights file. Only used for 'tsne' mode.
                        Default: None
  --top_db TOP_DB       Only used if --rosa is specified. Value under which
                        audio is considered as silence at beginning/end.
                        Default: 2
  --coins COINS [COINS ...]
                        Use only specified coin types. Possible values: 1, 2,
                        5, 20, 50, 100, 200. Default uses all coins.
  --num_examples NUM_EXAMPLES
                        Number of used coin data examples from each class for
                        training. Default uses the minimum number of all used
                        classes.
  --save_figures        Save figures of reconstructed time series.
  --seed SEED           Initializes Python, Numpy and Torch with this random
                        seed. !!NOTE: Before running the script export
                        PYTHONHASHSEED=0 as environment variable.!!
  --cudnn_deterministic
                        Sets CuDNN into deterministic mode. This might impact
                        perfromance.
  --no_state_dict       If set, saves the whole model instead of just the
                        weights.
  --run_cpu             If set, calculates on the CPU, even if GPU is
                        available.
  -ws WINDOW_SIZE, --window_size WINDOW_SIZE
                        Window size for training. Used if --use_windows is
                        specified. Default 1024.
  -wg WINDOW_GAP, --window_gap WINDOW_GAP
                        Gap between two consecutive windows. Default 1024.
  --use_windows         If set, training uses a sliding window with window
                        size specified by -ws. Default off, if using cnn
                        defaults to on.
  --save_plot SAVE_PLOT
                        Save file name for plots from 'tsne' and 'confusion'
                        modes. Default None
  --plot_title PLOT_TITLE
                        Title for 'tsne' and 'confusion' plots. Default None
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate. Default 0.001
  ```
