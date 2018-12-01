#!/usr/bin/env bash
# Chinese NER
# model: BiLSTM + CRF
# label:
# {"0", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"}
# reference: https://github.com/Determined22/zh-NER-TF

~/anaconda3/bin/python ./src/run.py \
--mode 'train' \
--train_data 'data_path' \
--test_data 'data_path' \
--batch_size 64 \
--epoch 2 \
--hidden_dim 300 \
--lr 0.001 \
--clip 5.0 \
--dropout 0.5 \
--update_embedding True \
--pretrain_embedding 'random' \
--embedding_dim 300 \
--shuffle True \
--demo_model '1521112368'

## mode: train/test
## clip: gradient clipping
## update_embedding: True/False, update embedding during training
## pretraining_embedding: use pretrained char embedding or init it randomly
## shuffle: shuffle training data before each epoch
## demo_model: model for test and demo, in data_path_save. while using your new trained model, change it.
