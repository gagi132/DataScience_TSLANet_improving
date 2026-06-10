# Mitigating Extrapolation Mismatch in Time-Series Forecasting via Structural Minimalism and Curvature Constraints

This repository contains the implementation of our framework designed to mitigate extrapolation mismatch in time-series forecasting through **Structural Minimalism (Strategy A)** and **Boundary-Selective Fisher Information Constraint (Strategy C)** built upon TSLANet.

---

## 📌 Reproducibility Resources

To facilitate reproducibility, we provide:

* `environment.yml` (recommended; reproduces the original experimental environment)
* `requirements.txt` (alternative pip-based installation)
* Google Drive dataset package (all benchmark datasets bundled in the required directory structure)
* Reproduction commands for all reported configurations

---

## 🚀 Quick Start (3 Steps)

### Step 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### Step 2. Set Up the Environment

Recommended:

```bash
conda env create -f environment.yml
conda activate tslanet
```

Alternative:

```bash
pip install -r requirements.txt
```

### Step 3. Download the Dataset Package

Download the dataset archive from:

```text
https://drive.google.com/drive/folders/1Fcw9C_juhXZi0wXPUFaXriyYtXKTicNL?usp=sharing
```

Extract the archive into the repository root directory so that the following structure is obtained:

```text
dataset/
├── ETT-small/
├── electricity/
├── weather/
├── exchange_rate/
└── METRLA/
```

---

## 🎯 Main Reproduction Command

The following command reproduces the primary METR-LA result reported in Table 1, where **Strategy A+C** achieves a Test MSE of **0.9852** and a Generalization Gap of **34.4%**.

```bash
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


## 🛠️ 1. Environment Setup

This project provides two installation methods. For exact reproducibility of the reported results, we strongly recommend using the provided Conda environment specification.

### Option A: Using Conda (Recommended)

Recreate the original experimental environment:

```bash
# Create the environment from the provided specification
conda env create -f environment.yml

# Activate the environment
conda activate tslanet
```

This method reproduces the software environment used in our experiments and is the recommended approach for obtaining results consistent with those reported in the paper.

### Option B: Using pip

Alternatively, install the required Python packages using:

```bash
pip install -r requirements.txt
```

This method provides a lightweight setup but may not perfectly reproduce the original environment due to differences in system-level dependencies, CUDA versions, or package resolution.

### Environment Verification

After installation, verify that the environment is correctly configured:

```bash
python --version
```

Expected output:

```text
Python 3.10.x
```

```bash
python -c "import torch; print(torch.__version__)"
```

If both commands execute successfully, the environment has been configured correctly.

```
```


## 📅 2. Data Preparation

All datasets used in our experiments are bundled and provided through a single Google Drive archive for ease of reproduction.

### Download

Download the dataset package from:

[Google Drive Link](https://drive.google.com/drive/folders/1Fcw9C_juhXZi0wXPUFaXriyYtXKTicNL?usp=sharing)

After downloading, extract the archive into the repository root directory.

Expected directory structure:

```text
dataset/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── electricity/
│   └── electricity.csv
├── weather/
│   └── weather.csv
├── exchange_rate/
│   └── exchange_rate.csv
└── METRLA/
    └── metr-la.h5
```

The provided file directory already follows the specified structure.

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
| **Strategy A + C** | **Our Finalized Framework (Linear + Boundary FIC)** | `--pretrain_mode timae_linear --use_fic True --fic_lambda 0.001 --fic_layerwise True` |
| **Strategy B + C** | Contrastive Pretraining + FIC | `--pretrain_mode contrastive --use_fic True --fic_lambda 0.001 --fic_layerwise True` |

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
