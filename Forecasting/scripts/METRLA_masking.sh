#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# METRLA_masking.sh  —  Baseline (original masking-MSE pretraining)
# ─────────────────────────────────────────────────────────────────────────────

for pred_len in 96 192 336 720
do
  echo "===== [MASKING] pred_len=${pred_len} ====="
  python -u ./Forecasting/TSLANet_Forecasting.py \
    --root_path    ./dataset/ \
    --data         METRLA \
    --seq_len      96 \
    --pred_len     ${pred_len} \
    --label_len    48 \
    --batch_size   32 \
    --train_epochs 20 \
    --pretrain_epochs 20 \
    --patch_size   16 \
    --emb_dim      64 \
    --depth        2 \
    --dropout      0.3 \
    --ICB          True \
    --ASB          True \
    --adaptive_filter True \
    --pretrain_mode   masking \
    --mask_ratio      0.4 \
    --load_from_pretrained True \
    --seed 42
done
