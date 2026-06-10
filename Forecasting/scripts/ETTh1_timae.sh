#!/bin/bash
# ETTh1 — Ti-MAE Pretraining Plugin
# Usage: bash ETTh1_timae.sh

for len in 96 192 336 720
do
  python -u TSLANet_Forecasting.py \
    --root_path C:/Emad/datasets/Forecasting/ETT-small \
    --pred_len $len \
    --data ETTh1 \
    --data_path ETTh1.csv \
    --seq_len 512 \
    --emb_dim 64 \
    --depth 1 \
    --batch_size 512 \
    --dropout 0.5 \
    --patch_size 32 \
    --train_epochs 20 \
    --pretrain_epochs 10 \
    --ASB False \
    \
    --pretrain_mode timae \
    --mask_ratio 0.4 \
    --timae_decoder_dim 64 \
    --timae_decoder_depth 2 \
    --timae_decoder_heads 4 \
    --timae_norm_pix_loss True
done
