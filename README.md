
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
Werkzeug==0.15.2
librosa==0.6.3
```

# Usage
```
torch_coin.py [-h] -p PATH [--use_variational_autoencoder] [-m MODE]
                     [-c CPU_COUNT] [-b BATCH_SIZE] [-lstm USE_LSTM]
                     [--val_split VAL_SPLIT] [-s SHRINK] [-hs HIDDEN_SIZE]
                     [-fc_hd FC_HIDDEN_DIM] [-e EPOCHS] [--save SAVE]
                     [-w WEIGHTS] [--top_db TOP_DB]
                     [--coins COINS [COINS ...]] [--num_examples NUM_EXAMPLES]
                     [--save_figures] [--seed SEED] [--cudnn_deterministic]

required arguments:
  -p PATH, --path PATH  Path to hdf5 data file.

optional arguments:
  -h, --help            show this help message and exit
  --use_variational_autoencoder
                        Uses a variational autoencoder model
  -m MODE, --mode MODE  Mode of the script. Can be either 'train', 'tsne',
                        'confusion' or 'infer'. Default 'train'
  -c CPU_COUNT, --cpu_count CPU_COUNT
                        Number of worker threads to use. Default 0
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size. Default 1
  -lstm USE_LSTM, --use_lstm USE_LSTM
                        Use lstm or gru. Default True = use lstm
  --val_split VAL_SPLIT
                        Validation split. Default is 0.1
  -s SHRINK, --shrink SHRINK
                        Shrinking factor. Selects data every s steps from
                        input.
  -hs HIDDEN_SIZE, --hidden_size HIDDEN_SIZE
                        Size of LSTM/GRU hidden layer.
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
```
