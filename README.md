# order_free_multi_label_classification

## Prerequisites

1. Python packages:
	- Python 3.5 or higher
	- Pytorch 1.0 or higher
	- Numpy
  - json
  - yaml
  
## Usage

1. Download data:

  - AAPD,RCV1: https://github.com/illcat/SGM-for-Multi-label-Classification
  
  - Reuters: Run data/preprocess_reuters.py
  
  - Move data to data/$dataset, which is a directory contains text_train, label_train.....

2. Preprocess data:

```
python3 data/preprocess.sh
```

3. Train model:

```
python3 train_rnn.py -gpus 0 -config config/config_$dataset.yaml
```

4. test model:

```
python3 train_rnn.py -gpus 0 -config config/config_$dataset.yaml -restore $expdir/best_in_train_micro_f1_checkpoint.pt -notrain
```

5. Hyperparameters can be modified in config/config_$dataset.yaml.

6. Logs can be found in exp directory.


