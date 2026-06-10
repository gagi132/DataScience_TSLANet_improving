# Mitigating Extrapolation Mismatch in Time-Series Forecasting via Structural Minimalism and Curvature Constraints

This repository contains the official implementation of our framework designed to mitigate the extrapolation mismatch in time-series forecasting using **Structural Minimalism (Strategy A)** and **Boundary-Selective Fisher Information Constraint (Strategy C)** built on top of TSLANet.

---

## 🚀 Quick Start & Reproducibility (TL;DR)

To reproduce the main METR-LA forecasting results (**Table 1** in the report, where Strategy A+C achieves a Test MSE of **0.9852** and a Generalization Gap of **34.4%**), execute the following single command after setting up the environment and data:

```bash
# Run the main reproduction script for Strategy A+C on METR-LA
python ./Forecasting/TSLANet_Forecasting.py \
    --root_path ./dataset/METRLA \
    --data METRLA \
    --data_path metr-la.h5 \
    --features M \
    --target OT \
    --seq_len 512 \
    --pred_len 96 \
    --batch_size 16 \
    --train_epochs 10 \
    --pretrain_epochs 20 \
    --patch_size 16 \
    --emb_dim 64 \
    --depth 2 \
    --pretrain_mode timae_linear \
    --timae_mask_ratio 0.75 \
    --use_fic True \
    --fic_lambda 0.001 \
    --fic_layerwise True \
    --seed 42

```

---

## 🛠️ 1. Environment Setup

This project is built using Python 3.10+ and PyTorch Lightning. We recommend using Anaconda for clean environment management.

### Option A: Using Conda (Recommended)

```bash
# Create and activate environment
conda create -n extrapolation python=3.10 -y
conda activate extrapolation

# Install PyTorch with CUDA support (adjust cuXXX to match your GPU hardware)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt

```

### Option B: Using pip

```bash
pip install -r requirements.txt

```

> **Note on `requirements.txt` Content:** Ensure your file contains at least: `pytorch-lightning`, `scikit-learn`, `pandas`, `numpy`, `h5py`, `einops`.

---

## 📅 2. Data Preparation

We evaluate our framework across 8 standard real-world benchmarks. Create a directory named `dataset/` at the repository root and arrange the files as follows:

```text
📁 dataset/
├── 📁 ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── 📁 electricity/
│   └── electricity.csv
├── 📁 weather/
│   └── weather.csv
├── 📁 exchange_rate/
│   └── exchange_rate.csv
└── 📁 METRLA/
    └── metr-la.h5

```

### Public Access Links:

* **ETT, Electricity, Weather, Exchange Rate:** Can be downloaded directly from the [Autoformer Repository](https://github.com/thuml/Autoformer) or [PatchTST Repository](https://github.com/yuqinie98/PatchTST).
* **METR-LA Spatio-Temporal Dataset:** Available via the [Graph WaveNet Repository](https://github.com/nnzhan/Graph-WaveNet) (download `metr-la.h5`).

---

## 📊 3. Reproducing Paper Benchmarks

We provide unified execution pathways to replicate all model variants discussed in our empirical analysis.

### Executing via Batch Script (`run.bat`)

For Windows environments, or to inspect the sequence of configurations for all benchmarks, you can run the pre-configured batch pipeline:

```bash
# This will execute the comprehensive evaluation pipeline sequentially
run.bat

```

### Configuration Mapping Reference

| Strategy Profile | Description | Core Flags |
| --- | --- | --- |
| **Original Baseline** | Vanilla TSLANet with Masking-MSE | `--pretrain_mode masking --use_fic False` |
| **Ti-MAE (Heavy)** | Multi-layer Transformer Decoder | `--pretrain_mode timae --use_fic False` |
| **Strategy A** | Structural Minimalist Linear Decoder | `--pretrain_mode timae_linear --use_fic False` |
| **Strategy B** | Contrastive InfoNCE Pretaining | `--pretrain_mode contrastive --use_fic False` |
| **Strategy A + C** | **Our Finalized Framework (Linear + Boundary FIC)** | `--pretrain_mode timae_linear --use_fic True --fic_lambda 0.001` |
| **Strategy B + C** | Contrastive Pretraining + FIC | `--pretrain_mode contrastive --use_fic True --fic_lambda 0.001` |

---

## 📁 4. Project Structure

```text
├── 📁 Dataset/             # Data loaders and preprocessing scripts
│   ├── metr_la_dataset.py  # METR-LA h5 streaming and channel processing
│   └── ts_dataset.py       # Standard time-series data parsing
├── 📁 Models/              # Neural Network Architectures
│   ├── TSLANet.py          # Temporal Backbone Core
│   └── plugins.py          # Decoders (Transformer Decoder / TiMAELinearPlugin)
├── 📁 Optimization/        # Curvature regularizers
│   └── fic.py              # Fisher Information Constraint (FIC) core implementation
├── 📄 TSLANet_Forecasting.py # Main train/validation/test pipeline executable
├── 📄 requirements.txt     # Python environment package specification
└── 📄 run.bat              # Batch execution configuration script for reproduction

```

---

## 🔍 Verification of Training Trajectory

Upon execution, PyTorch Lightning will automatically generate verification folders inside `lightning_logs/`. The console log outputs metrics sequentially, ending with an evaluation summary confirming the stable suppression of the generalization gap:

```text
Testing DataLoader 0: 100%|█████████████████████████████████████████████████████████|
MSE: {'test': 0.98520, 'val': 0.73312}
MAE: {'test': 0.61264, 'val': 0.52110}

```
