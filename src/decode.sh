#!/bin/bash

python annotate.py --data_dir=../data --train_dir=../models --lstm_size=1024 --num_layers=3 --encoder_vocab_size=150000 --decoder_vocab_size=3878 --batch_size=128 --name=test --decode=True
