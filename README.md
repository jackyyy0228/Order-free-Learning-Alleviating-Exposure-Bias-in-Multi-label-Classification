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
	bash data/preprocess.sh
	```

	- This script contains the codes for preprocessing the three datasets.
  
  	- The script will transcibe the labels into vector format.
  
  	- The order of labels is from frequent to rare.
	
	- Modify the path of dataset in data/preprocess.sh.

3. Train model:

	```
	python3 train_rnn.py -gpus 0 -config config/config_$dataset.yaml
	```

	- Hyperparameters can be modified in config/config_$dataset.yaml
	
	- Log can be found in the log directory.
	
	```
	python3 train_logistic_baseline.py -gpus 0 -config config/config_$dataset.yaml
	```

	- Codes for training binary relevance model.

4. test model:

	```
	python3 train_rnn.py -gpus 0 -config config/config_$dataset.yaml -restore $expdir/best_in_train_micro_f1_checkpoint.pt -notrain
	```


## Files and directories

`models` : codes for model structure

`metrics.py` : metrics for multi-label classification

`optims.py` : code for optimizer and defining gradient clipping.

`preprocess.py` : code for preprocessing

## Hyperparameters

`data`:  The path of file, save_data. e.g. './data/AAPD/save_data'

`epoch`: Number  of epoch for training.

`train_batch_size`: Batch size for training.

`test_batch_size`: Batch size for testing for beam search.

`log`:  Directory for log files. e.g. './exp/aapd/ocd'

`emb_size`: Size of word embedding.

`load_emb`: Load pretrained word vectors. (We set false for random initialization)

`emb_path`: Path of pretrained word embedding. (if load_emb is true)

`hidden_size`: Hidden size for LSTM cell.

`encoder_n_layers`: Number of layers for LSTM encoder.

`decoder_n_layers`: Number of layers for LSTM decoder.

`input_dropout_p`: Probability of dropout for input of encoder.

`dropout_p`: Probability of dropout for RNN encoder and decoder.

`bidirectional`: BLSTM or not.

`logistic_weight`: weight of loss between BR decoder and RNN decoder.

`max_tgt_len`: maximun number of decoding steps.

`loss_type`: vallina, OCD  ,or order_free

`beam_size`: beam size for decoding

`add_mask`: mask to prevent rnn decoder generate same labels.

`OCD_temperature_start`: Start temperature for OCD.

`OCD_temperature_end`: End temperature for OCD.

`OCD_final_hard_epoch`: Number of epoches for ocd temperature to reach OCD_temperature_end (Linear decay).

`eval_interval`: Number of updates to check the performance in validation set.

`print_interval`: Number of updates to print the current average loss.
