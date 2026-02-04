# E-MAS Quick Start Guide

## 1. Installation (5 minutes)

```bash
# Clone and setup
cd emas-skin-lesion-classifier
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Prepare Data

Download datasets:
- **HAM10000**: [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) → `data/ham10000/`
- **PH2**: [PH2 Database](https://www.fc.up.pt/addi/ph2%20database.html) → `data/ph2/`

## 3. Launch Web App

```bash
streamlit run app.py
```

Open http://localhost:8501

## 4. Train Model (Optional)

```bash
python train.py \
    --dataset ham10000 \
    --data-dir data/ham10000 \
    --ham-csv data/ham10000/HAM10000_metadata.csv \
    --epochs 40 \
    --batch-size 32
```

## 5. Evaluate Model

```bash
python evaluate.py \
    --checkpoint checkpoints/emas_ham10000_best_*.pth \
    --dataset ham10000 \
    --data-dir data/ham10000
```

## 6. Deploy with Docker

```bash
docker build -t emas-app .
docker run -p 8501:8501 emas-app
```

---

**Full documentation**: See [README.md](README.md)  
**Deployment guide**: See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
