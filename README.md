# ♻️ Waste Classification using ResNet50

Phan loai rac thai (Huu co vs Tai che) su dung mo hinh ResNet50 voi PyTorch va Transfer Learning.

---

## 📋 Bai toan

| Muc | Mo ta |
|-----|-------|
| **Nhiem vu** | Phan loai anh rac thai thanh 2 nhan: Huu co (O) va Tai che (R) |
| **Mo hinh** | ResNet50 (pretrained ImageNet) + Fine-tuning |
| **Framework** | PyTorch + CUDA |
| **Ket qua** | Accuracy: 0.96 · AUC: 0.99 · F1: 0.96 |

---

## 📁 Cau truc thu muc

```
waste-classification/
├── data/
│   └── DATASET/
│       ├── TRAIN/
│       │   ├── O/          # 12.565 anh Huu co
│       │   └── R/          # 9.999 anh Tai che
│       └── TEST/
│           ├── O/          # 1.401 anh Huu co
│           └── R/          # 1.112 anh Tai che
├── notebooks/
│   └── waste_classification.ipynb
├── models/
│   ├── resnet50_waste.pth   # Trong so mo hinh (~90 MB)
│   └── class_mapping.json
├── outputs/
│   ├── samples.png
│   ├── distribution.png
│   ├── class_distribution.png
│   ├── eval_curves.png
│   ├── f1_curve.png
│   └── predictions.png
├── app/
│   └── app.py              # Streamlit demo
├── kaggle.json             # Kaggle API key (khong push len GitHub)
├── requirements.txt
└── README.md
```

---

## 📊 Bo du lieu

- **Ten**: [Waste Classification Data](https://www.kaggle.com/datasets/techsash/waste-classification-data)
- **Nguon**: Kaggle – techsash
- **Tong so anh**: ~25.100 anh
- **Chia tap**: 85% TRAIN (22.564 anh) / 15% TEST (2.513 anh)
- **Nhan**:
  - `O` → Organic (Huu co): 12.565 train / 1.401 test
  - `R` → Recyclable (Tai che): 9.999 train / 1.112 test

---

## ⚙️ Yeu cau he thong

| Thanh phan | Phien ban |
|-----------|-----------|
| Python | 3.11 |
| CUDA | 12.1+ |
| GPU | NVIDIA (khuyen nghi >= 4 GB VRAM) |
| OS | Windows 10/11 |

---

## 🚀 Huong dan cai dat

### 1. Clone repository

```bash
git clone https://github.com/<your-username>/waste-classification.git
cd waste-classification
```

### 2. Tao moi truong ao

```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Cai dat thu vien

```cmd
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4. Cai dat Jupyter kernel

```cmd
python -m ipykernel install --user --name=waste-venv --display-name "Python (waste-venv)"
```

---

## 📥 Cai dat Dataset

### Cach 1: Tu dong qua Kaggle API

1. Tao tai khoan Kaggle tai [kaggle.com](https://www.kaggle.com)
2. Vao **Settings → API → Create New Token** → tai file `kaggle.json`
3. Dat file `kaggle.json` vao thu muc goc du an:

```
waste-classification/kaggle.json
```

4. Chay cell sau trong notebook:

```python
import os
os.environ["KAGGLE_CONFIG_DIR"] = r"<duong_dan_den_thu_muc_du_an>"

import kaggle
kaggle.api.authenticate()
kaggle.api.dataset_download_files(
    "techsash/waste-classification-data",
    path=r"<duong_dan_den_thu_muc_du_an>\data",
    unzip=True
)
```

### Cach 2: Thu cong

Tai dataset tai: https://www.kaggle.com/datasets/techsash/waste-classification-data  
Giai nen vao `data/DATASET/` sao cho co cau truc:

```
data/DATASET/TRAIN/O/
data/DATASET/TRAIN/R/
data/DATASET/TEST/O/
data/DATASET/TEST/R/
```

---

## 🏋️ Huan luyen mo hinh

Mo file `notebooks/waste_classification.ipynb` trong VSCode, chon kernel **Python (waste-venv)**, chay tung cell theo thu tu:

| Cell | Noi dung |
|------|----------|
| Cell 0 | Kiem tra GPU |
| Cell 1 | Doc dataset tu Kaggle |
| Cell 2 | In cau truc folder |
| Cell 2.1 | Tinh chinh duong dan TRAIN/TEST |
| Cell 3 | Kiem tra so luong anh va class |
| Cell 4 | Kham pha du lieu (bieu do, anh mau) |
| Cell 5 | Tien xu ly + DataLoader |
| Cell 6 | Cau hinh sieu tham so |
| Cell 7 | Kien truc mo hinh + Early Stopping |
| Cell 8 | Huan luyen mo hinh |
| Cell 9 | Danh gia (accuracy, confusion matrix, curves) |
| Cell 9.1 | Du doan ngau nhien 10 anh |
| Cell 10 | Luu mo hinh |

---

## 🎛️ Sieu tham so

| Tham so | Gia tri |
|---------|---------|
| `model` | resnet50 (pretrained ImageNet) |
| `img_size` | 224 × 224 |
| `batch_size` | 32 |
| `epochs` | 30 |
| `learning_rate` | 1e-4 (freeze) / 1e-5 (unfreeze) |
| `weight_decay` | 1e-4 |
| `optimizer` | AdamW |
| `scheduler` | CosineAnnealingLR |
| `T_max` | 30 |
| `eta_min` | 1e-6 |
| `early_stopping` | patience = 5 |
| `unfreeze_epoch` | 5 (mo dong toan bo sau epoch 5) |
| `dropout` | 0.5 |

### Chien luoc huan luyen (Transfer Learning 2 giai doan)

- **Giai doan 1 (epoch 1–4)**: Dong bang toan bo backbone, chi huan luyen lop `fc`
- **Giai doan 2 (epoch 5+)**: Mo dong toan bo, fine-tune voi lr = 1e-5

---

## 📈 Ket qua

| Chi so | Gia tri |
|--------|---------|
| **Test Accuracy** | 0.96 |
| **Test AUC** | 0.99 |
| **Test F1 (weighted)** | 0.96|

### Classification Report

```
              precision    recall  f1-score   support

     Organic       0.96      0.99      0.97      1401
  Recyclable       0.98      0.95      0.96      1112

    accuracy                           0.97      2513
   macro avg       0.97      0.97      0.97      2513
weighted avg       0.97      0.97      0.97      2513
```

### Bieu do ket qua

Cac bieu do duoc luu tu dong vao `outputs/`:

| File | Noi dung |
|------|----------|
| `class_distribution.png` | Phan bo nhan TRAIN/TEST |
| `samples.png` | Anh mau 2 class |
| `distribution.png` | Phan bo kich thuoc anh |
| `eval_curves.png` | Loss / Accuracy / AUC qua tung epoch |
| `f1_curve.png` | F1-Score qua tung epoch |
| `predictions.png` | Du doan ngau nhien 10 anh |

---

## 🖥️ Demo Streamlit

### Tinh nang

- Upload anh rac thai (JPG/PNG)
- Hien thi anh goc + **Grad-CAM heatmap** (vung mo hinh chu y)
- Ket qua: nhan du doan (Huu co / Tai che) + xac suat
- Bieu do cot xac suat 2 class

### Chay app

```cmd
cd "duong_dan_den_thu_muc_du_an"
venv\Scripts\activate
streamlit run app/app.py
```

Truy cap tai: http://localhost:8501

---

## 📦 requirements.txt

```
torch==2.3.1+cu121
torchvision==0.18.1+cu121
torchaudio==2.3.1+cu121
jupyter
ipykernel
numpy<2.0
pandas
matplotlib
seaborn
scikit-learn
Pillow
kaggle
tqdm
grad-cam
streamlit
opencv-python==4.9.0.80
```

## 👤 Tac gia

Franceto (ANH PHAP TO)

---

## 📄 Giay phep

MIT License
