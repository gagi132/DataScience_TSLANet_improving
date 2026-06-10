@echo off
chcp 437 > nul

set ROOT=./dataset/ETT-small
set DATA=ETTm1
set FILE=ETTm1.csv

set SEQ=512
set BS_N=32
set BS_C=16

set EMB=64
set DEPTH=2
set PS=16

set TR=20
set PT=20


echo [MODE 1] Masking-MSE Baseline
python -u ./Forecasting/TSLANet_Forecasting.py ^
    --root_path %ROOT% --data %DATA% ^
    --data_path %FILE% ^
    --features M ^
    --target OT ^
    --seq_len %SEQ% --pred_len 96 ^
    --batch_size %BS_N% ^
    --train_epochs %TR% --pretrain_epochs %PT% ^
    --patch_size %PS% --emb_dim %EMB% --depth %DEPTH% ^
    --pretrain_mode masking --mask_ratio 0.4 ^
    --use_fic False --seed 42

echo [MODE 2] Ti-MAE Transformer Decoder
python -u ./Forecasting/TSLANet_Forecasting.py ^
    --root_path %ROOT% --data %DATA% ^
    --data_path %FILE% ^
    --features M ^
    --target OT ^
    --seq_len %SEQ% --pred_len 96 ^
    --batch_size %BS_N% ^
    --train_epochs %TR% --pretrain_epochs %PT% ^
    --patch_size %PS% --emb_dim %EMB% --depth %DEPTH% ^
    --pretrain_mode timae ^
    --timae_mask_ratio 0.75 ^
    --timae_decoder_dim 64 ^
    --timae_decoder_depth 2 ^
    --timae_decoder_heads 4 ^
    --use_fic False --seed 42

echo [MODE 3] Ti-MAE Linear Decoder
python -u ./Forecasting/TSLANet_Forecasting.py ^
    --root_path %ROOT% --data %DATA% ^
    --data_path %FILE% ^
    --features M ^
    --target OT ^
    --seq_len %SEQ% --pred_len 96 ^
    --batch_size %BS_N% ^
    --train_epochs %TR% --pretrain_epochs %PT% ^
    --patch_size %PS% --emb_dim %EMB% --depth %DEPTH% ^
    --pretrain_mode timae_linear ^
    --timae_mask_ratio 0.75 ^
    --use_fic False --seed 42

echo [MODE 4] Strategy B - Contrastive InfoNCE
python -u ./Forecasting/TSLANet_Forecasting.py ^
    --root_path %ROOT% ^
    --data %DATA% ^
    --data_path %FILE% ^
    --features M ^
    --target OT ^
    --seq_len %SEQ% ^
    --pred_len 96 ^
    --batch_size %BS_C% ^
    --train_epochs %TR% ^
    --pretrain_epochs 10 ^
    --patch_size %PS% ^
    --emb_dim %EMB% ^
    --depth %DEPTH% ^
    --pretrain_mode contrastive ^
    --contrastive_temperature 0.2 ^
    --projection_dim 128 ^
    --context_weight 1.0 ^
    --temporal_weight 1.0 ^
    --aug_weight 1.0 ^
    --augmentations jitter,scaling ^
    --use_fic False ^
    --seed 42

echo [MODE 5] Ti-MAE Linear Decoder + FIC
python -u ./Forecasting/TSLANet_Forecasting.py ^
    --root_path %ROOT% --data %DATA% ^
    --data_path %FILE% ^
    --features M ^
    --target OT ^
    --seq_len %SEQ% --pred_len 96 ^
    --batch_size %BS_N% ^
    --train_epochs %TR% --pretrain_epochs %PT% ^
    --patch_size %PS% --emb_dim %EMB% --depth %DEPTH% ^
    --pretrain_mode timae_linear ^
    --timae_mask_ratio 0.75 ^
    --use_fic True ^
    --fic_lambda 0.001 ^
    --seed 42

echo [MODE 6] Strategy B+C - Contrastive + FIC
python -u ./Forecasting/TSLANet_Forecasting.py ^
    --root_path %ROOT% ^
    --data %DATA% ^
    --data_path %FILE% ^
    --features M ^
    --target OT ^
    --seq_len %SEQ% ^
    --pred_len 96 ^
    --batch_size %BS_C% ^
    --train_epochs %TR% ^
    --pretrain_epochs 10 ^
    --patch_size %PS% ^
    --emb_dim %EMB% ^
    --depth %DEPTH% ^
    --pretrain_mode contrastive ^
    --contrastive_temperature 0.2 ^
    --projection_dim 128 ^
    --context_weight 1.0 ^
    --temporal_weight 1.0 ^
    --aug_weight 1.0 ^
    --augmentations jitter,scaling ^
    --use_fic True ^
    --fic_lambda 0.001 ^
    --fic_layerwise True ^
    --seed 42




set ROOT=./dataset/METRLA
set DATA=METRLA
set SEQ=512
set BS_N=16
set BS_C=8
set EMB=64
set DEPTH=2
set PS=16
set TR=20
set PT=10


echo [MODE 1] Masking-MSE Baseline
python -u ./Forecasting/TSLANet_Forecasting.py ^
    --root_path %ROOT% --data %DATA% ^
    --data_path %FILE% ^
    --features M ^
    --target OT ^
    --seq_len %SEQ% --pred_len 96 ^
    --batch_size %BS_N% ^
    --train_epochs %TR% --pretrain_epochs %PT% ^
    --patch_size %PS% --emb_dim %EMB% --depth %DEPTH% ^
    --pretrain_mode masking --mask_ratio 0.4 ^
    --use_fic False --seed 42

echo [MODE 2] Ti-MAE Transformer Decoder
python -u ./Forecasting/TSLANet_Forecasting.py ^
    --root_path %ROOT% --data %DATA% ^
    --data_path %FILE% ^
    --features M ^
    --target OT ^
    --seq_len %SEQ% --pred_len 96 ^
    --batch_size %BS_N% ^
    --train_epochs %TR% --pretrain_epochs %PT% ^
    --patch_size %PS% --emb_dim %EMB% --depth %DEPTH% ^
    --pretrain_mode timae ^
    --timae_mask_ratio 0.75 ^
    --timae_decoder_dim 64 ^
    --timae_decoder_depth 2 ^
    --timae_decoder_heads 4 ^
    --use_fic False --seed 42

echo [MODE 3] Ti-MAE Linear Decoder
python -u ./Forecasting/TSLANet_Forecasting.py ^
    --root_path %ROOT% --data %DATA% ^
    --data_path %FILE% ^
    --features M ^
    --target OT ^
    --seq_len %SEQ% --pred_len 96 ^
    --batch_size %BS_N% ^
    --train_epochs %TR% --pretrain_epochs %PT% ^
    --patch_size %PS% --emb_dim %EMB% --depth %DEPTH% ^
    --pretrain_mode timae_linear ^
    --timae_mask_ratio 0.75 ^
    --use_fic False --seed 42

echo [MODE 4] Strategy B - Contrastive InfoNCE
python -u ./Forecasting/TSLANet_Forecasting.py ^
    --root_path %ROOT% ^
    --data %DATA% ^ 
    --data_path metr-la.h5 ^
    --features M ^
    --target OT ^
    --seq_len %SEQ% --pred_len 96 ^
    --batch_size %BS_C% ^
    --train_epochs %TR% --pretrain_epochs 10 ^
    --patch_size %PS% --emb_dim %EMB% --depth %DEPTH% ^
    --pretrain_mode contrastive ^
    --contrastive_temperature 0.2 --projection_dim 128 ^
    --context_weight 1.0 --temporal_weight 1.0 --aug_weight 1.0 ^
    --augmentations jitter,scaling ^
    --use_fic False --seed 42

echo [MODE 5] Ti-MAE Linear Decoder + FIC
python -u ./Forecasting/TSLANet_Forecasting.py ^
    --root_path %ROOT% --data %DATA% ^
    --data_path %FILE% ^
    --features M ^
    --target OT ^
    --seq_len %SEQ% --pred_len 96 ^
    --batch_size %BS_N% ^
    --train_epochs %TR% --pretrain_epochs %PT% ^
    --patch_size %PS% --emb_dim %EMB% --depth %DEPTH% ^
    --pretrain_mode timae_linear ^
    --timae_mask_ratio 0.75 ^
    --use_fic True ^
    --fic_lambda 0.001 ^
    --seed 42

echo [MODE 6] Strategy B+C - Contrastive + FIC
python -u ./Forecasting/TSLANet_Forecasting.py ^
    --root_path %ROOT% --data %DATA% ^ --data_path metr-la.h5 ^
    --features M ^
    --target OT ^
    --seq_len %SEQ% --pred_len 96 ^
    --batch_size %BS_C% ^
    --train_epochs %TR% --pretrain_epochs 10 ^
    --patch_size %PS% --emb_dim %EMB% --depth %DEPTH% ^
    --pretrain_mode contrastive ^
    --contrastive_temperature 0.2 --projection_dim 128 ^
    --context_weight 1.0 --temporal_weight 1.0 --aug_weight 1.0 ^
    --augmentations jitter,scaling ^
    --use_fic True --fic_lambda 0.001 --fic_finetune False --fic_layerwise False ^
    --seed 42

echo All experiments done.
pause
