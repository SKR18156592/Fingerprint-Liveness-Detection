# Fingerprint Liveness Detection (Presentation Attack Detection)

A deep learning-based **Presentation Attack Detection (PAD)** system that classifies fingerprint images as **LIVE** or **SPOOF** using **Transfer Learning** with **MobileNetV3-Small** and TensorFlow/Keras.

---

## Overview

Presentation Attack Detection (PAD) is a critical security layer in biometric authentication systems. Before fingerprint matching is performed, the PAD module determines whether the captured fingerprint originates from a **real finger** or from a **presentation attack**, such as:

* Printed fingerprint photos
* Displayed fingerprint images on phone screens
* Other spoof artifacts

This project implements a binary classifier that distinguishes between **Live** and **Spoof** fingerprints and calibrates its operating threshold using biometric evaluation metrics such as **BPCER**, **APCER**, **ACER**, and **Equal Error Rate (EER)**.

---

## Features

* Transfer Learning using **MobileNetV3-Small**
* Binary Fingerprint Classification
* Automatic Dataset Split (70/15/15)
* Image Augmentation
* ImageNet Normalization
* Threshold Calibration using Validation Set
* APCER, BPCER and ACER computation
* Equal Error Rate (EER)
* ROC Curve
* Score Distribution Plot
* APCER–BPCER Tradeoff Curve
* Confusion Matrix
* Single Image Inference
* Automatic Metric Export (CSV)

---

## Project Structure

```text
Fingerprint-Liveness-Detection/
│
├── data/
│   ├── live/
│   ├── spoof/
│   └── raw/
│
├── dataset_split/
│   ├── train/
│   │   ├── live/
│   │   └── spoof/
│   ├── val/
│   │   ├── live/
│   │   └── spoof/
│   └── test/
│       ├── live/
│       └── spoof/
│
├── models/
│   └── liveness_model.keras
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── score_distribution.png
│   ├── apcer_bpcer_tradeoff.png
│   ├── evaluation_metrics.csv
│   ├── best_threshold.txt
│   └── training_log.csv
│
├── test_images/
│
├── dataset_split.py
├── dataset.py
├── liveness_train.py
├── liveness_eval.py
├── liveness_infer.py
├── requirements.txt
└── README.md

```

---

## Dataset

### Directory Structure

```text
data/
├── live/
└── spoof/

```

### Dataset Split & Summary

| Split | Percentage | Live Count | Spoof Count | Total Count |
| --- | --- | --- | --- | --- |
| Training | 70% | 84 | 84 | 168 |
| Validation | 15% | 18 | 18 | 36 |
| Testing | 15% | 18 | 18 | 36 |

All images are resized to **224 × 224** pixels with 3 color channels before training.

---

## Model Architecture

* **Backbone:** MobileNetV3-Small (Fine-tuning the last 25 layers)
* **Framework:** TensorFlow / Keras
* **Input Shape:** `(None, 224, 224, 3)`
* **Global Average Pooling 2D:** `(None, 576)`
* **Dropout Layer:** `0.3` rate
* **Sigmoid Output Layer:** `Dense(1, activation="sigmoid")`
* **Parameters:** 939,697 total (~3.58 MB) | 490,129 trainable (~1.87 MB)

---

## Training Configuration

| Parameter | Value |
| --- | --- |
| Optimizer | Adam |
| Learning Rate | 0.0001 (`1e-4`) |
| Epochs | 30 (Early Stopping) |
| Image Size | 224 × 224 |
| Batch Size | 8 |
| Loss Function | Binary Crossentropy |
| Callbacks | ModelCheckpoint, EarlyStopping (patience=6), ReduceLROnPlateau (factor=0.2, patience=3, min_lr=1e-6), CSVLogger |

---

## Data Augmentation

Training images undergo:

* Random Rotation
* Random Horizontal Flip
* Random Brightness
* Random Contrast

---

## Evaluation Metrics & Performance Benchmarks

The model is evaluated using rigorous biometric and classification metrics:

* **Test Accuracy:** **97.22%**
* **ROC-AUC Score:** **1.000**
* **F1-Score:** **0.9730**
* **Precision:** **0.9474**
* **Recall / True Positive Rate:** **1.0000**
* **APCER (Attack Presentation Classification Error Rate):** **0.0000** (Zero false-negative spoof bypasses)
* **BPCER (Bona Fide Presentation Classification Error Rate):** **0.0556**
* **ACER (Average Classification Error Rate):** **0.0278**
* **Equal Error Rate (EER):** **0.0000**
* **Calibrated Operating Threshold:** **0.3300** (Targeting a BPCER of ~3%)

Threshold calibration is performed on the **validation set** by sweeping thresholds from **0.00 to 1.00** with a step size of **0.01**. The operating threshold is selected where **BPCER ≈ 3%**, and the model is then evaluated on the **test set** using this calibrated threshold.

---

## Generated Outputs

The evaluation script generates:

* Confusion Matrix (`confusion_matrix.png`)
* ROC Curve (`roc_curve.png`)
* Score Distribution (`score_distribution.png`)
* APCER–BPCER Tradeoff Curve (`apcer_bpcer_tradeoff.png`)
* Evaluation Metrics CSV (`evaluation_metrics.csv`)
* Best Threshold File (`best_threshold.txt`)

---

## Running the Project

### 1. Clone Repository

```bash
git clone [https://github.com/SKR18156592/Fingerprint-Liveness-Detection.git](https://github.com/SKR18156592/Fingerprint-Liveness-Detection.git)

cd Fingerprint-Liveness-Detection

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Prepare Dataset

Place fingerprint images inside:

```text
data/
├── live/
└── spoof/

```

Generate dataset splits:

```bash
python dataset_split.py

```

### 4. Train the Model

```bash
python liveness_train.py

```

The trained model will be saved to:

```text
models/liveness_model.keras

```

### 5. Evaluate the Model

```bash
python liveness_eval.py

```

Generated outputs will be saved in:

```text
outputs/

```

### 6. Run Inference

Update the image path inside `liveness_infer.py` or place your image at:

```python
IMAGE_PATH = Path("test_images/sample_fingerprint.jpg")

```

Then run:

```bash
python liveness_infer.py

```

Example output:

```text
Loading model...
Model loaded.
Loaded calibrated threshold: 0.3300
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 440ms/step

Prediction
----------------------
Label      : SPOOF ❌
Score      : 0.9984
Confidence : 99.84%

```

---

## Results Summary Table

| Metric | Score / Value |
| --- | --- |
| **Threshold (BPCER≈3%)** | `0.3300` |
| **Precision** | `0.9474` |
| **Recall** | `1.0000` |
| **F1 Score** | `0.9730` |
| **APCER** | `0.0000` |
| **BPCER** | `0.0556` |
| **ACER** | `0.0278` |
| **EER** | `0.0000` |
| **ROC-AUC** | `1.0000` |
| **Test Accuracy** | `97.22%` |

---

## Future Improvements

Given more time and a larger dataset, future work could include:

* Larger and more diverse datasets
* Advanced spoof attack types
* Cross-device evaluation
* Real-time webcam inference
* Model quantization for mobile deployment
* Vision Transformer or EfficientNet backbones
* Domain adaptation for unseen sensors

---

## Requirements

* Python 3.10+
* TensorFlow
* OpenCV
* NumPy
* Matplotlib
* Pandas
* Scikit-learn

Install using:

```bash
pip install -r requirements.txt

```

---

## License

This project was developed for academic purposes as part of a Fingerprint Presentation Attack Detection assignment.

```

```