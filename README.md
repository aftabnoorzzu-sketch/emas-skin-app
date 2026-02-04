# E-MAS: Efficient Multi-Scale Attention System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**E-MAS** is a deep learning framework for automated dermoscopic skin lesion classification, combining MobileNetV2 and EfficientNet-B0 with ASPP and SE attention mechanisms.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Web Application](#web-application)
- [Docker Deployment](#docker-deployment)
- [Project Structure](#project-structure)
- [Results](#results)
- [Citation](#citation)

## ✨ Features

- **Dual Backbone Architecture**: Combines MobileNetV2 and EfficientNet-B0
- **Point-wise Feature Fusion**: Element-wise multiplication for complementary features
- **ASPP Module**: Multi-scale contextual feature extraction with dilation rates [6, 12, 18]
- **SE Attention**: Channel-wise adaptive recalibration
- **Grad-CAM Explainability**: Visualize model attention on input images
- **Multiple Datasets**: Support for HAM10000, PH2, and Combined datasets
- **Stratified Splits**: 70/15/15 train/val/test with 5-fold CV option
- **Interactive Web UI**: Streamlit-based interface for inference and training

## 🏗️ Architecture

```
Input (224×224×3)
    │
    ├──► MobileNetV2 ──┐
    │                    ├──► Point-wise Multiplication ─► ASPP ─► SE ─► GAP ─► Classifier
    └──► EfficientNet-B0 ┘
```

### Key Components

1. **Feature Extraction**: MobileNetV2 + EfficientNet-B0 (ImageNet pretrained)
2. **Feature Fusion**: F_fused = F_mobilenet ⊙ F_efficientnet
3. **ASPP**: Atrous convolutions with rates [6, 12, 18] + global pooling
4. **SE Block**: Squeeze-and-Excitation attention (reduction=16)
5. **Classifier**: GAP → Dense(256→128) → Dense(128→num_classes) → Softmax

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional, CPU supported)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd emas-skin-lesion-classifier
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 📁 Dataset Preparation

### HAM10000 Dataset

1. Download from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
2. Extract to `data/ham10000/`:

```
data/ham10000/
├── HAM10000_images_part_1/
├── HAM10000_images_part_2/
└── HAM10000_metadata.csv
```

### PH2 Dataset

1. Download from [PH2 Database](https://www.fc.up.pt/addi/ph2%20database.html)
2. Extract to `data/ph2/` with class subdirectories:

```
data/ph2/
├── nev/
├── atypical/
└── mel/
```

### Folder Structure Option

Alternatively, organize HAM10000 by class:

```
data/ham10000/
├── akiec/
├── bcc/
├── bkl/
├── df/
├── mel/
├── nv/
└── vasc/
```

## 🚀 Usage

### Quick Start - Web Application

Launch the Streamlit interface:

```bash
streamlit run app.py
```

Access at: http://localhost:8501

### Web App Features

- **Inference Tab**: Upload images and get predictions with confidence scores
- **Grad-CAM Visualization**: See which regions the model focuses on
- **Train Tab**: Configure and launch training jobs
- **Evaluate Tab**: Generate comprehensive evaluation reports
- **About Model**: View architecture details and performance metrics

## 🏋️ Training

### Basic Training (HAM10000)

```bash
python train.py \
    --dataset ham10000 \
    --data-dir data/ham10000 \
    --ham-csv data/ham10000/HAM10000_metadata.csv \
    --epochs 40 \
    --batch-size 32 \
    --lr 0.001
```

### Training (PH2)

```bash
python train.py \
    --dataset ph2 \
    --data-dir data/ph2 \
    --epochs 40 \
    --batch-size 16 \
    --lr 0.001
```

### Training (Combined Dataset)

```bash
python train.py \
    --dataset combined \
    --data-dir '{"ham10000": "data/ham10000", "ph2": "data/ph2"}' \
    --ham-csv data/ham10000/HAM10000_metadata.csv \
    --epochs 40 \
    --batch-size 32 \
    --lr 0.001
```

### 5-Fold Cross-Validation

```bash
python train.py \
    --dataset ham10000 \
    --data-dir data/ham10000 \
    --split-mode 5fold \
    --fold-idx 0 \
    --epochs 40
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | required | Dataset: ham10000, ph2, combined |
| `--data-dir` | required | Path to dataset directory |
| `--ham-csv` | None | Path to HAM10000 metadata CSV |
| `--epochs` | 40 | Number of training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--weight-decay` | 1e-4 | Weight decay |
| `--split-mode` | holdout | Split mode: holdout or 5fold |
| `--fold-idx` | None | Fold index for 5-fold CV |
| `--checkpoint-dir` | checkpoints | Directory to save checkpoints |
| `--early-stopping-patience` | 7 | Early stopping patience |

## 📊 Evaluation

### Evaluate Model

```bash
python evaluate.py \
    --checkpoint checkpoints/emas_ham10000_best_YYYYMMDD_HHMMSS.pth \
    --dataset ham10000 \
    --data-dir data/ham10000 \
    --ham-csv data/ham10000/HAM10000_metadata.csv \
    --batch-size 32
```

### Evaluation Outputs

Reports are saved to `reports/`:

- `evaluation_report_{dataset}_{timestamp}.json` - Detailed metrics
- `evaluation_report_{dataset}_{timestamp}.csv` - Summary table
- `confusion_matrix_{dataset}_{timestamp}.png` - Confusion matrix plot
- `roc_curves_{dataset}_{timestamp}.png` - ROC curves plot

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t emas-skin-lesion-classifier .
```

### Run Container

```bash
docker run -p 8501:8501 \
    -v $(pwd)/checkpoints:/app/checkpoints \
    -v $(pwd)/data:/app/data \
    emas-skin-lesion-classifier
```

### Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  emas-app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./reports:/app/reports
      - ./data:/app/data
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

Run:

```bash
docker-compose up
```

## 📁 Project Structure

```
emas-skin-lesion-classifier/
├── app.py                  # Streamlit web application
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
├── README.md               # This file
├── models/
│   ├── __init__.py
│   └── emas.py            # E-MAS model architecture
├── data/
│   ├── __init__.py
│   └── datasets.py        # Dataset loaders
├── utils/
│   ├── __init__.py
│   ├── gradcam.py         # Grad-CAM implementation
│   └── preprocess.py      # Image preprocessing
├── checkpoints/           # Model checkpoints (created at runtime)
└── reports/               # Evaluation reports (created at runtime)
```

## 📈 Results

### Performance Metrics

| Dataset | Accuracy | AUC | Sensitivity | Precision | Specificity |
|---------|----------|-----|-------------|-----------|-------------|
| HAM10000 | 98.73% | 98.21% | 95.12% | 94.37% | 99.78% |
| PH2 | 97.78% | 99.01% | 98.80% | 97.10% | 99.37% |
| Combined | 98.20% | 99.00% | 96.40% | 97.10% | 99.20% |

### HAM10000 Classes

| Code | Full Name | Description |
|------|-----------|-------------|
| akiec | Actinic keratoses | Pre-cancerous skin lesions |
| bcc | Basal cell carcinoma | Most common skin cancer |
| bkl | Benign keratosis-like | Non-cancerous lesions |
| df | Dermatofibroma | Benign skin nodules |
| mel | Melanoma | Serious skin cancer |
| nv | Melanocytic nevi | Common moles |
| vasc | Vascular lesions | Blood vessel abnormalities |

### PH2 Classes

| Code | Full Name |
|------|-----------|
| nev | Nevus |
| atypical | Atypical nevus |
| mel | Melanoma |

## 🔧 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'models'`

**Solution**: Ensure you're running from the project root directory.

---

**Issue**: `CUDA out of memory`

**Solution**: Reduce batch size or use CPU:
```python
device = torch.device('cpu')
```

---

**Issue**: `FileNotFoundError: [Errno 2] No such file or directory`

**Solution**: Verify dataset paths and ensure data is extracted correctly.

---

**Issue**: Grad-CAM not showing

**Solution**: Ensure model is loaded and image is preprocessed correctly.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{emas2024,
  title={E-MAS: An Efficient Multi-Scale Attention System for Dermoscopic 
         Image-Based Skin Cancer Classification},
  author={Aftab, Muhammad and Ali, Muhammad Mumtaz and Dong, Zigang and 
          Zhang, Chengjuan and Zhenfei, Wang and Jiang, Yanan and Liu, Kangdong},
  journal={},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- HAM10000 dataset: [Tschandl et al., Scientific Data 2018](https://www.nature.com/articles/sdata2018161)
- PH2 dataset: [Mendonça et al., EMBC 2013](https://www.fc.up.pt/addi/ph2%20database.html)
- PyTorch and torchvision teams

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the authors.

---

**Disclaimer**: This tool is for research purposes only and should not be used as a substitute for professional medical diagnosis. Always consult a qualified dermatologist for skin lesion evaluation.
