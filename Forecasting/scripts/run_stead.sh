#!/bin/bash
# STEAD Earthquake Forecasting Test for TSLANet

python -u ./Forecasting/TSLANet_Forecasting.py \
  --root_path ./dataset/ \
  --data STEAD \
  --seq_len 2000 \
  --pred_len 500 \
  --emb_dim 64 \
  --depth 3 \
  --batch_size 32 \
  --dropout 0.5 \
  --patch_size 16 \
  --train_epochs 20 \
  --pretrain_epochs 10