# 🛰️ SatOrbit — Model Training

This branch contains the **machine learning and deep learning development** for the SatOrbit satellite image change-detection system.

The main model used in this branch is a **Siamese U-Net** trained for **pixel-level binary change detection** using the **Onera Satellite Change Detection (OSCD)** dataset.

The trained model will later be integrated into the SatOrbit backend for detecting and analyzing changes in real Sentinel-2 satellite imagery.

---

# 🎯 Project Objective

The objective is to compare satellite images of the **same geographical area at two different points in time** and identify which pixels have changed.

The system is designed to detect changes related to:

* 🏗️ New construction
* 🌱 Vegetation loss
* 💧 Surface-water changes
* 🏞️ Other land-cover changes

The deep-learning model performs **binary change detection** first.

```text
0 → No Change
1 → Change
```

The type of change is determined in a **separate analysis stage** using spectral information and change-region analysis.

---

# 🧠 Overall SatOrbit Pipeline

```text
              Satellite Image T1
                     +
              Satellite Image T2
                     │
                     ▼
              Data Preprocessing
                     │
                     ▼
             Image Registration
                     │
                     ▼
               Image Pairs
                     │
                     ▼
               Siamese U-Net
                     │
                     ▼
            Binary Change Mask
              0 = No Change
              1 = Change
                     │
                     ▼
           Changed Region Extraction
                     │
                     ▼
          Change-Type Classification
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
      Vegetation    Water   Construction
                     │
                     ▼
              Area Calculation
                     │
                     ▼
                 results.json
                     │
                     ▼
             SatOrbit Backend
                     │
                     ▼
               Frontend Map
```

---

# 🧠 Why Siamese U-Net?

Normal U-Net is mainly designed for segmentation of a single image.

Change detection requires comparison between **two temporal images**.

Siamese U-Net is designed for this purpose.

```text
              Image T1                 Image T2
             Earlier Date              Later Date
                  │                         │
                  ▼                         ▼
           ┌─────────────┐           ┌─────────────┐
           │   Encoder   │           │   Encoder   │
           │             │           │             │
           │ Shared      │           │ Shared      │
           │ Weights     │           │ Weights     │
           └──────┬──────┘           └──────┬──────┘
                  │                         │
                  ▼                         ▼
             Features T1              Features T2
                  │                         │
                  └──────────┬──────────────┘
                             │
                             ▼
                    Feature Difference
                         |F1 - F2|
                             │
                             ▼
                       U-Net Decoder
                             │
                             ▼
                    Change Probability
                             │
                             ▼
                       Change Mask
```

The two images use the **same encoder weights**, allowing the model to learn comparable features from both dates.

---

# 🛰️ Dataset

The primary dataset used for model development is:

**Onera Satellite Change Detection (OSCD)**

OSCD contains multi-temporal satellite imagery and corresponding ground-truth change labels.

Dataset:

[OSCD Dataset — IEEE DataPort](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection?utm_source=chatgpt.com)

---

# 📥 Dataset Setup

The OSCD dataset must be downloaded separately.

**Do not upload the dataset to GitHub.**

Place the extracted dataset inside:

```text
data/
└── OSCD/
```

The local directory may contain folders such as:

```text
data/
└── OSCD/
    ├── Onera Satellite Change Detection dataset - Images/
    ├── Onera Satellite Change Detection dataset - Train Labels/
    └── Onera Satellite Change Detection dataset - Test Labels/
```

> The exact internal structure may vary depending on the downloaded OSCD version. The preprocessing script should be configured according to the actual local dataset structure.

---

# ⚠️ Important: Do Not Commit Dataset Files

Large satellite datasets must remain local.

The following should not be committed:

```text
data/
*.zip
*.tif
*.tiff
```

The `.gitignore` file should contain the required exclusions.

---

# 📂 Project Structure

The recommended structure of this branch is:

```text
model_training/
│
├── data/
│   └── OSCD/
│       └── local dataset
│
├── notebooks/
│   ├── data_analysis.ipynb
│   ├── preprocessing.ipynb
│   └── model_training.ipynb
│
├── scripts/
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│   └── trained model weights
│
├── outputs/
│   ├── predictions/
│   ├── evaluation/
│   └── plots/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🔄 Complete Model-Training Workflow

Every team member should follow this workflow.

```text
1. Download OSCD
        ↓
2. Place dataset in data/OSCD/
        ↓
3. Create Python environment
        ↓
4. Install requirements
        ↓
5. Explore dataset
        ↓
6. Preprocess images
        ↓
7. Create temporal image pairs
        ↓
8. Prepare ground-truth masks
        ↓
9. Generate training patches
        ↓
10. Train Siamese U-Net
        ↓
11. Validate model
        ↓
12. Evaluate on test data
        ↓
13. Save trained model
        ↓
14. Generate change masks
        ↓
15. Perform change-type analysis
```

---

# 1️⃣ Clone the Repository

Clone the repository:

```powershell
git clone <repository-url>
```

Move into the project:

```powershell
cd SatOrbit-Prototype
```

Switch to the model-training branch:

```powershell
git checkout model_training
```

Pull the latest changes:

```powershell
git pull origin model_training
```

---

# 2️⃣ Create Python Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\Activate.ps1
```

Verify Python:

```powershell
python --version
```

---

# 3️⃣ Install Dependencies

Install the required Python packages:

```powershell
pip install -r requirements.txt
```

The project uses technologies including:

* Python
* NumPy
* Pandas
* OpenCV
* Pillow
* Rasterio
* PyTorch
* Torchvision
* scikit-learn
* Matplotlib

---

# 4️⃣ Verify Dataset

Before training, verify that the OSCD dataset is available locally.

Example:

```text
data/
└── OSCD/
```

The dataset should contain:

```text
Satellite Images
+
Ground Truth Labels
```

Do not start training if the dataset paths are incorrect.

---

# 5️⃣ Dataset Exploration

Before preprocessing, inspect the dataset.

Use:

```text
notebooks/data_analysis.ipynb
```

The analysis should verify:

* Number of locations
* Image dimensions
* Available bands
* Temporal image pairs
* Label dimensions
* Label values
* Missing/corrupt files
* Distribution of changed and unchanged pixels

Example:

```text
Image T1
    │
    ├── Dimensions
    ├── Bands
    └── Metadata

Image T2
    │
    ├── Dimensions
    ├── Bands
    └── Metadata

Ground Truth
    │
    ├── Dimensions
    └── Change labels
```

---

# 6️⃣ Data Preprocessing

Run:

```powershell
python scripts/preprocessing.py
```

The preprocessing pipeline should perform the required operations such as:

1. Read satellite images
2. Identify temporal image pairs
3. Select required bands
4. Check dimensions
5. Align/co-register images when required
6. Normalize image values
7. Process ground-truth labels
8. Generate corresponding image pairs
9. Generate training patches
10. Apply appropriate augmentation where required

---

# 7️⃣ Image Pair Creation

The model requires two images representing the same geographical area at different times.

```text
Image T1
Earlier
   +
Image T2
Later
   │
   ▼
Training Sample
```

For example:

```text
T1 → Earlier satellite image
T2 → Later satellite image
GT → Ground-truth change mask
```

The model learns:

```text
(T1, T2) → Change Mask
```

---

# 8️⃣ Ground Truth

OSCD ground-truth labels are used as the target for supervised training.

For the initial model:

```text
0 → No Change
1 → Change
```

Example:

```text
Ground Truth

0 0 0 0 0
0 0 1 1 0
0 1 1 1 0
0 0 1 0 0
0 0 0 0 0
```

The Siamese U-Net attempts to predict a mask similar to this.

---

# 9️⃣ Patch Generation

Large satellite images can be divided into smaller patches.

The initial target patch size is:

```text
256 × 256
```

For every patch, the following must correspond spatially:

```text
T1 Patch
+
T2 Patch
+
Ground Truth Patch
```

Example:

```text
patch_0001

T1 → patch_0001
T2 → patch_0001
GT → patch_0001
```

All three must represent the same geographical region.

---

# 📁 Processed Dataset

After preprocessing, the recommended structure is:

```text
data/
└── processed/
    │
    ├── train/
    │   ├── A/
    │   ├── B/
    │   └── labels/
    │
    ├── val/
    │   ├── A/
    │   ├── B/
    │   └── labels/
    │
    └── test/
        ├── A/
        ├── B/
        └── labels/
```

Example:

```text
train/
├── A/
│   ├── patch_0001.tif
│   ├── patch_0002.tif
│   └── patch_0003.tif
│
├── B/
│   ├── patch_0001.tif
│   ├── patch_0002.tif
│   └── patch_0003.tif
│
└── labels/
    ├── patch_0001.png
    ├── patch_0002.png
    └── patch_0003.png
```

---

# 🔟 Train / Validation / Test

The dataset should be divided into:

```text
Training Data
       │
       ├── Model learns
       │
       ▼
Validation Data
       │
       ├── Tune and monitor
       │
       ▼
Test Data
       │
       └── Final evaluation
```

The test data should **not** be used for model training.

Avoid data leakage between training and test locations/regions.

---

# 1️⃣1️⃣ Siamese U-Net Training

Run:

```powershell
python scripts/train.py
```

The training process is:

```text
             Image T1
                 │
                 ▼
              Encoder
                 │
                 ▼
                F1
                 │
                 │
                 │ Feature Difference
                 ▼
              |F1-F2|
                 ▲
                 │
                F2
                 ▲
                 │
              Encoder
                 ▲
                 │
             Image T2
                 │
                 ▼
              Decoder
                 │
                 ▼
         Predicted Change Mask
                 │
                 ▼
        Compare with Ground Truth
                 │
                 ▼
                Loss
                 │
                 ▼
        Backpropagation
                 │
                 ▼
          Update Weights
```

This process is repeated for multiple batches and epochs.

---

# 📉 Loss Function

The initial loss function is:

```text
Total Loss = BCE Loss + Dice Loss
```

### Binary Cross Entropy

Helps the model classify individual pixels as:

```text
0 → No Change
1 → Change
```

### Dice Loss

Helps improve overlap between predicted and ground-truth change regions.

This is particularly useful because changed pixels may represent a relatively small portion of the image.

---

# ⚙️ Initial Model Configuration

| Parameter      | Configuration                    |
| -------------- | -------------------------------- |
| Architecture   | Siamese U-Net                    |
| Framework      | PyTorch                          |
| Input          | Two temporal satellite images    |
| Patch Size     | 256 × 256                        |
| Task           | Binary Change Detection          |
| Classes        | 2                                |
| No Change      | 0                                |
| Change         | 1                                |
| Encoder        | Shared CNN Encoder               |
| Feature Fusion | Absolute Difference              |
| Loss           | BCE + Dice                       |
| Evaluation     | IoU, Dice, F1, Precision, Recall |

These parameters can be changed during experimentation.

---

# 1️⃣2️⃣ Model Evaluation

Run:

```powershell
python scripts/evaluate.py
```

The model should be evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* IoU
* Dice coefficient

For segmentation/change detection, the most important metrics are:

```text
IoU
Dice Score
F1 Score
Precision
Recall
```

---

# 📊 Evaluation Concept

```text
Ground Truth Mask
        +
Predicted Mask
        │
        ▼
   Pixel Comparison
        │
        ▼
┌───────┼────────┐
▼       ▼        ▼
 IoU   Dice      F1
```

The final model should not be selected based only on accuracy.

---

# 1️⃣3️⃣ Model Output

After training, save the model weights inside:

```text
models/
```

Example:

```text
models/
└── siamese_unet_change_detection.pth
```

The model checkpoint may contain:

* Model weights
* Optimizer state
* Epoch
* Training loss
* Validation loss
* Best metric

---

# 1️⃣4️⃣ Prediction

After training, the model receives:

```text
Image T1 + Image T2
```

and produces:

```text
Change Probability Map
        ↓
Threshold
        ↓
Binary Change Mask
```

Example:

```text
Input:
T1 + T2

Output:

0 0 0 1 1
0 0 1 1 1
0 0 0 0 0
1 1 1 0 0
```

where:

```text
0 → No Change
1 → Change
```

---

# 🏗️ Change-Type Classification

## Important

The Siamese U-Net initially performs **binary change detection**.

It answers:

> **WHERE did change occur?**

It does not directly answer:

> **WHAT type of change occurred?**

Therefore, SatOrbit uses a second analysis stage.

```text
                 Siamese U-Net
                      │
                      ▼
              Binary Change Mask
                      │
                      ▼
             Changed Regions
                      │
                      ▼
             Spectral Analysis
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
         NDVI        NDWI        NDBI
          │           │           │
          ▼           ▼           ▼
      Vegetation     Water    Built-up /
        Change       Change   Construction
```

---

# 🌱 Vegetation Change

Vegetation change can be analyzed using **NDVI** and the detected change mask.

NDVI:

```text
NDVI = (NIR - Red) / (NIR + Red)
```

A significant decrease in NDVI inside a detected change region can indicate vegetation loss.

Conceptually:

```text
High NDVI T1
       ↓
Low NDVI T2
       ↓
Detected Change
       ↓
Possible Vegetation Loss
```

---

# 💧 Water Change

Water-related changes can be analyzed using **NDWI**.

NDWI:

```text
NDWI = (Green - NIR) / (Green + NIR)
```

Examples:

```text
Water present in T1
Water absent in T2
       ↓
Water Loss
```

or:

```text
Water absent in T1
Water present in T2
       ↓
New / Increased Water
```

---

# 🏗️ Construction Change

Construction can be analyzed using:

* Detected change mask
* Spectral information
* Built-up indices such as NDBI
* Spatial characteristics
* Land-cover information

A typical pattern is:

```text
Vegetated / Bare Region in T1
            ↓
Built-up Region in T2
            ↓
Persistent Changed Region
            ↓
Possible New Construction
```

Construction classification should be treated as an additional analysis layer rather than assuming that every changed pixel is construction.

---

# 📐 Area Calculation

Once changed regions are identified, their physical area can be calculated.

Basic formula:

```text
Area = Number of Pixels × Area per Pixel
```

For example:

```text
Changed pixels
      ×
Ground area represented by each pixel
      =
Changed area
```

For different categories:

```text
Vegetation Loss Area
= Vegetation-change pixels × Pixel Area

Water Change Area
= Water-change pixels × Pixel Area

Construction Area
= Construction-change pixels × Pixel Area
```

The final values can be converted to:

```text
m²
hectares
km²
```

depending on the application requirements.

---

# 📄 Example SatOrbit Result

The final analysis can produce a JSON structure similar to:

```json
{
  "area_analyzed_km2": 33.48,
  "total_changed_area_km2": 3.85,
  "change_rate_percent": 11.51,
  "changes": {
    "new_construction_km2": 2.68,
    "vegetation_loss_km2": 0.42,
    "water_change_km2": 0.75
  }
}
```

> These values are examples only. Actual values must come from model predictions and spatial analysis.

---

# 📁 Output Structure

Training and evaluation outputs should be organized as:

```text
outputs/
│
├── predictions/
│   ├── predicted_mask_001.png
│   ├── predicted_mask_002.png
│   └── ...
│
├── evaluation/
│   ├── metrics.json
│   └── evaluation_report.txt
│
└── plots/
    ├── training_loss.png
    ├── validation_loss.png
    └── metrics.png
```

---

# 🧪 Recommended Experiment Tracking

Every team member should record important training experiments.

Example:

```text
Experiment: 001

Model:
Siamese U-Net

Patch Size:
256 × 256

Epochs:
50

Batch Size:
8

Learning Rate:
0.001

Loss:
BCE + Dice

Best Validation IoU:
...

Best Validation Dice:
...

Test IoU:
...

Test Dice:
...
```

This prevents the team from losing track of which configuration produced the best model.

---

# 💻 GPU / CPU

Training a deep-learning model is significantly faster with a GPU.

Check PyTorch CUDA availability:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

If the output is:

```text
True
```

a CUDA-compatible GPU is available to PyTorch.

If:

```text
False
```

the model can still run on CPU, but training may take significantly longer.

---

# 🔧 Common Problems

## Dataset Not Found

Check:

```text
data/OSCD/
```

and verify the dataset path configured in the preprocessing script.

---

## CUDA Not Available

Check:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

If CUDA is unavailable, training can run on CPU or on another configured GPU environment.

---

## Out of Memory

Reduce:

```text
Batch Size
```

or:

```text
Patch Size
```

and retry training.

---

## Incorrect Image/Label Dimensions

Before training, verify:

```text
T1 dimensions
=
T2 dimensions
=
Ground Truth dimensions
```

for every training sample.

---

# 🔐 Git & Large Files

Do not commit:

```text
data/
*.zip
*.tif
*.tiff
*.pth
*.pt
```

Check the repository before pushing:

```powershell
git status
```

Add only source code, notebooks, configuration files, documentation, and other appropriate small files.

---

# 🔄 Git Workflow for Team Members

Before starting work:

```powershell
git checkout model_training
git pull origin model_training
```

After making changes:

```powershell
git status
```

Add required files:

```powershell
git add .
```

Commit:

```powershell
git commit -m "Update model training pipeline"
```

Push:

```powershell
git push origin model_training
```

Avoid committing datasets and large model files.

---

# 🧑‍💻 Recommended Team Responsibilities

The model-training work can be divided into:

```text
Member 1
│
├── Dataset exploration
└── OSCD analysis

Member 2
│
├── Preprocessing
├── Image alignment
└── Patch generation

Member 3
│
├── Siamese U-Net
└── Model architecture

Member 4
│
├── Training
├── Hyperparameter tuning
└── Experiment tracking

Member 5
│
├── Evaluation
├── Visualization
└── Change-type analysis
```

Team members can work independently while following the same folder structure and preprocessing rules.

---

# 🚀 Final SatOrbit Integration

Once the best Siamese U-Net model has been trained and evaluated, it will be integrated into the main SatOrbit application.

```text
              model_training
                    │
                    │
              Trained Model
                    │
                    ▼
              SatOrbit Backend
                    │
                    ▼
          Sentinel-2 Image Pair
                    │
                    ▼
              Siamese U-Net
                    │
                    ▼
             Change Mask
                    │
                    ▼
          Change-Type Analysis
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
      Vegetation   Water   Construction
          │         │         │
          └─────────┼─────────┘
                    ▼
              Area Calculation
                    │
                    ▼
                results.json
                    │
                    ▼
              Frontend Map
```

---

# 🔬 Current Development Status

```text
🟡 Model Development
```

Current objectives:

* [x] Select OSCD dataset
* [x] Select Siamese U-Net architecture
* [ ] Complete dataset exploration
* [ ] Implement preprocessing
* [ ] Generate image pairs
* [ ] Prepare ground-truth masks
* [ ] Generate training patches
* [ ] Implement Siamese U-Net
* [ ] Train baseline model
* [ ] Evaluate model
* [ ] Improve model performance
* [ ] Implement change-type analysis
* [ ] Calculate affected areas
* [ ] Integrate trained model with SatOrbit backend

---

# 🔮 Future Improvements

Possible future improvements include:

* Multi-band Sentinel-2 input
* Improved image co-registration
* Data augmentation
* Hyperparameter tuning
* Improved loss functions
* U-Net++
* Attention U-Net
* Transformer-based change detection
* Multi-class change detection
* Improved construction detection
* Improved vegetation and water classification
* Real-world Sentinel-2 testing
* Model optimization for inference
* Backend integration

---

# ⚠️ Important Model Design Decision

The current system intentionally uses a **two-stage approach**:

```text
Stage 1
Siamese U-Net
        ↓
Binary Change Detection
        ↓
WHERE did change happen?


Stage 2
Spectral / Spatial Analysis
        ↓
Change Classification
        ↓
WHAT changed?
        ↓
Vegetation / Water / Construction
```

This design is used because the OSCD dataset primarily provides **change/no-change ground truth**, rather than reliable class labels for construction, vegetation loss, and water changes.

Therefore, the project should **not claim that the Siamese U-Net itself directly classifies every change type**.

---

# 📌 Quick Start

For a team member who wants to train the model from scratch:

```powershell
# 1. Clone repository
git clone <repository-url>

# 2. Enter project
cd SatOrbit-Prototype

# 3. Switch branch
git checkout model_training

# 4. Get latest code
git pull origin model_training

# 5. Create environment
python -m venv venv

# 6. Activate environment
venv\Scripts\Activate.ps1

# 7. Install dependencies
pip install -r requirements.txt

# 8. Download OSCD separately
# Place it inside:
# data/OSCD/

# 9. Explore dataset
jupyter notebook

# 10. Preprocess
python scripts/preprocessing.py

# 11. Train
python scripts/train.py

# 12. Evaluate
python scripts/evaluate.py
```

After successful training:

```text
models/
└── siamese_unet_change_detection.pth
```

The trained model can then be used for SatOrbit inference and backend integration.

---

# 👥 Project Information

**Project:** SatOrbit

**SIH Problem Statement:** SIH 227

**Branch:** `model_training`

**Primary Dataset:** Onera Satellite Change Detection (OSCD)

**Model:** Siamese U-Net

**Task:** Pixel-Level Binary Change Detection

**Framework:** PyTorch

**Change Detection:** Deep Learning

**Change-Type Analysis:** Spectral and spatial analysis

**Target Categories:**

* 🏗️ New Construction
* 🌱 Vegetation Loss
* 💧 Surface-Water Changes
* 🏞️ Other Land-Cover Changes

**Purpose:** Develop, train, evaluate, and prepare the deep-learning change-detection model for integration into the SatOrbit satellite imagery analysis system.
