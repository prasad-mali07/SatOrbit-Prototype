# 🛰️ SatOrbit — Model Training

This branch contains the **machine learning and deep learning model-training work** for the SatOrbit satellite image change-detection project.

The purpose of this branch is to experiment with satellite imagery, prepare training data, train change-detection/segmentation models, and evaluate their performance.

---

## 🎯 Objective

The main objective is to develop a deep-learning model capable of detecting and segmenting changes in satellite imagery captured at different points in time.

The model is intended to identify changes such as:

* 🏗️ New construction
* 🌱 Vegetation loss
* 💧 Surface-water changes
* 🏞️ Other land-cover changes

---

## 🧠 Model Training Pipeline

```text
Satellite Dataset
       │
       ▼
Data Preprocessing
       │
       ▼
Image Pair Creation
       │
       ▼
Ground Truth / Labels
       │
       ▼
Train / Validation / Test Split
       │
       ▼
Model Training
       │
       ▼
Model Evaluation
       │
       ▼
Trained Model
       │
       ▼
Change Detection
```

---

## 📂 Branch Structure

```text
model_training/
│
├── notebooks/
│   ├── data_analysis.ipynb
│   ├── preprocessing.ipynb
│   └── model_training.ipynb
│
├── scripts/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│   └── trained models
│
├── outputs/
│   ├── predictions/
│   └── evaluation/
│
├── data/
│   └── Local dataset
│
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** Large satellite datasets are intentionally excluded from GitHub using `.gitignore`. The `data/` directory should remain local.

---

## 🛰️ Dataset

The project uses satellite imagery for change-detection experiments.

The primary dataset used for model-development experiments is the **Onera Satellite Change Detection (OSCD)** dataset.

The dataset contains multi-temporal satellite images and corresponding change labels for different geographical locations.

### Dataset Structure

The local dataset may contain:

```text
https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection
data/
└── OSCD/
    ├── Onera Satellite Change Detection dataset - Images/
    ├── Onera Satellite Change Detection dataset - Train Labels/
    └── Onera Satellite Change Detection dataset - Test Labels/
```

The dataset is **not uploaded to GitHub** because of its large size.

---

## 🔧 Technologies

### Programming

* Python
* NumPy
* Pandas

### Image Processing

* OpenCV
* Pillow
* Rasterio

### Machine Learning / Deep Learning

* PyTorch
* scikit-learn
* U-Net
* CNN-based segmentation

### Visualization

* Matplotlib

---

## 🧹 Data Preprocessing

Before training, satellite images are processed to make them suitable for the model.

Typical preprocessing steps include:

1. Reading satellite images
2. Selecting required bands
3. Image alignment
4. Resizing
5. Normalization
6. Creating image pairs
7. Processing ground-truth masks
8. Creating training patches

Example:

```text
Image at T1 ──────┐
                  ├──► Preprocessing ──► Training Pair
Image at T2 ──────┘

Ground Truth ────────────────► Target Mask
```

---

## 🏷️ Ground Truth

Ground-truth masks are used to teach the model which pixels have changed.

A simplified representation:

```text
0 → No Change
1 → Change
```

The model learns to predict a change mask from two satellite images.

---

## 🤖 U-Net Approach

The planned deep-learning architecture is **U-Net**, which is commonly used for image segmentation.

```text
        Input Images
             │
             ▼
      ┌─────────────┐
      │  Encoder    │
      └──────┬──────┘
             │
             ▼
      ┌─────────────┐
      │ Bottleneck  │
      └──────┬──────┘
             │
             ▼
      ┌─────────────┐
      │  Decoder    │
      └──────┬──────┘
             │
             ▼
       Change Mask
```

The objective is to predict a pixel-level change map.

---

## 📊 Model Evaluation

The trained model can be evaluated using metrics such as:

* Accuracy
* Precision
* Recall
* F1-score
* IoU (Intersection over Union)
* Dice coefficient

For segmentation, **IoU and Dice score** are particularly useful for measuring overlap between predicted and ground-truth change regions.

---

## 🏋️ Training

Create and activate a Python virtual environment:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the training script:

```powershell
python scripts/train.py
```

If training is performed through a Jupyter Notebook:

```powershell
jupyter notebook
```

Then open the training notebook from:

```text
notebooks/
```

---

## 💾 Model Output

After training, the model weights can be saved in:

```text
models/
```

For example:

```text
models/
└── unet_change_detection.pth
```

Large model files may also be excluded from Git depending on their size.

---

## 🔬 Current Status

```text
🟡 Model Training / Development
```

Current work includes:

* Dataset exploration
* Satellite-image preprocessing
* Ground-truth preparation
* Image-pair generation
* Change-mask analysis
* Deep-learning model development
* Model evaluation

---

## 🔮 Future Work

* Improve image preprocessing and alignment
* Train U-Net on the OSCD dataset
* Experiment with different loss functions
* Improve IoU and Dice scores
* Perform data augmentation
* Experiment with U-Net++
* Compare CNN and transformer-based segmentation models
* Integrate the trained model into the SatOrbit backend
* Perform real-world Sentinel-2 change detection

---

## 🔗 Integration with SatOrbit

The trained model from this branch is intended to become part of the main SatOrbit change-detection pipeline.

```text
model_training
       │
       │ Trained Model
       ▼
SatOrbit Backend
       │
       ▼
Satellite Image Change Detection
       │
       ▼
Frontend Visualization
```

---

## ⚠️ Dataset & Large Files

The following types of files should **not** be committed to GitHub:

```text
data/
*.zip
*.tif
*.tiff
*.pth
*.pt
```

Use `.gitignore` to keep large datasets and model files out of Git history.

The dataset should be downloaded separately and placed inside the local `data/` directory.

---

## 👥 Project

**SatOrbit**

Satellite Image Change Detection System

**Branch:** `model_training`

**Purpose:** Machine Learning / Deep Learning development for satellite change detection.
