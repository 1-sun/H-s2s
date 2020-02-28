#!/bin/bash

python annotate.py --data_dir=../data --train_dir=../models --lstm_size=1024 --num_layers=3 --batch_size=64 --encoder_vocab_size=150000 --decoder_vocab_size=3878 --learning_rate=0.1 --dropout=0.2 --name=test
