#!/bin/bash
python -u ./Forecasting/TSLANet_Forecasting.py \
  --root_path ./dataset/ \
  --data METRLA \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 207 \
  --dec_in 207 \
  --c_out 207 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 20