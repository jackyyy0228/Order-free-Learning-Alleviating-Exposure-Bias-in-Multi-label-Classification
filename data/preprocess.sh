#!/bin/bash

## AAPD
python3 preprocess.py -train_src ./data/AAPD/text_train \
  -train_tgt ./data/AAPD/label_train \
  -valid_src ./data/AAPD/text_val \
  -valid_tgt ./data/AAPD/label_val \
  -test_src ./data/AAPD/text_test \
  -test_tgt ./data/AAPD/label_test \
  -save_data ./data/AAPD/save_data \
  -src_vocab_size 30000

## Reuters 
python3 data/preprocess_reuters.py ./data/reuters
python3 data/sort_label.py data/reuters/label_train data/reuters/label_val data/reuters/label_test
python3 preprocess.py -train_src ./data/reuters/text_train \
  -train_tgt ./data/reuters/label_train \
  -valid_src ./data/reuters/text_val \
  -valid_tgt ./data/reuters/label_val \
  -test_src ./data/reuters/text_test \
  -test_tgt ./data/reuters/label_test \
  -save_data ./data/reuters/save_data \
  -fasttext_model  ./data/wiki_model.bin \
  -embedding ./data/reuters/embedding.pt \
  -src_vocab_size 22747
python3 data/sort_label.py data/AAPD/label_train data/AAPD/label_val data/AAPD/label_test


##RCV1-V2
python3 preprocess.py -train_src ./data/RCV1-V2/text_train \
  -train_tgt ./data/RCV1-V2/label_train \
  -valid_src ./data/RCV1-V2/text_val \
  -valid_tgt ./data/RCV1-V2/label_val \
  -test_src ./data/RCV1-V2/text_test \
  -test_tgt ./data/RCV1-V2/label_test \
  -save_data ./data/RCV1-V2/save_data \
  -src_vocab_size 50000
##AAPD
  
