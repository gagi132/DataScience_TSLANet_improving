#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# METRLA_timae.sh  —  Ti-MAE pretraining plugin
# ─────────────────────────────────────────────────────────────────────────────

for pred_len in 96 192 336 720
do
  echo "===== [Ti-MAE] pred_len=${pred_len} ====="
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
    --pretrain_mode      timae \
    --timae_mask_ratio   0.75 \
    --timae_decoder_dim  64 \
    --timae_decoder_depth 2 \
    --timae_decoder_heads 4 \
    --timae_norm_pix_loss True \
    --load_from_pretrained True \
    --seed 42
done
