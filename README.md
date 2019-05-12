
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
```

# Usage
```
torch_coin.py [-h] [-m MODE] [-c CPU_COUNT] [-b BATCH_SIZE]
                     [-lstm USE_LSTM] [-l LOG_FILE] [-p PATH] [-s SHRINK]
                     [-hs HIDDEN_SIZE] [-id IDENTIFIER] [-d] [-e EPOCHS]
                     [--save SAVE] [-w WEIGHTS]

optional arguments:
  -h, --help            show this help message and exit
  -m MODE, --mode MODE  Mode of the script. Can be either 'train', ''tsne' or
                        infer'. Default 'train'
  -c CPU_COUNT, --cpu_count CPU_COUNT
                        Number of cpus to use. Default 0
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size. Default 1
  -lstm USE_LSTM, --use_lstm USE_LSTM
                        Use lstm or gru. Default True = use lstm
  -l LOG_FILE, --log_file LOG_FILE
                        CSV logfile. Creates path if it does not exist.
                        Default 'metrics.csv'
  -p PATH, --path PATH  Path to working directory, used as base dataset path
                        and base log file path. Default ./
  -s SHRINK, --shrink SHRINK
                        Shrinking factor. Selects data every s steps from
                        input.
  -hs HIDDEN_SIZE, --hidden_size HIDDEN_SIZE
                        Size of LSTM/GRU hidden layer.
  -id IDENTIFIER, --identifier IDENTIFIER
                        Unique identifier for the current run.
  -d, --debug
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs
  --save SAVE           Specify save folder for weight files. Default: None
  -w WEIGHTS, --weights WEIGHTS
                        Model weights file. Only used for 'tsne' mode.
                        Default: None
```
