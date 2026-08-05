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

# Dataset

## Directory Structure

```
data/
├── live/
└── spoof/
```

### Dataset Split

| Split      | Percentage |
| ---------- | ---------: |
| Training   |        70% |
| Validation |        15% |
| Testing    |        15% |

All images are resized to **224 × 224** pixels before training.

---

# Model Architecture

* Backbone: **MobileNetV3-Small**
* Framework: TensorFlow / Keras
* Transfer Learning
* Global Average Pooling
* Dropout (0.3)
* Sigmoid Output Layer

---

# Training Configuration

| Parameter     | Value               |
| ------------- | ------------------- |
| Optimizer     | Adam                |
| Epochs        | 20                  |
| Image Size    | 224 × 224           |
| Batch Size    | 8                   |
| Loss Function | Binary Crossentropy |

> **Note:** If following the assignment specification exactly, use a learning rate of **0.001**.

---

# Data Augmentation

Training images undergo:

* Random Rotation
* Random Horizontal Flip
* Random Brightness
* Random Contrast

---

# Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* APCER
* BPCER
* ACER
* Equal Error Rate (EER)
* ROC-AUC

Threshold calibration is performed on the **validation set** by sweeping thresholds from **0.00 to 1.00** with a step size of **0.01**. The operating threshold is selected where **BPCER ≈ 3%**, and the model is then evaluated on the **test set** using this calibrated threshold.

---

# Generated Outputs

The evaluation script generates:

* Confusion Matrix
* ROC Curve
* Score Distribution
* APCER–BPCER Tradeoff Curve
* Evaluation Metrics CSV
* Best Threshold File

---

# Running the Project

## 1. Clone Repository

```bash
git clone https://github.com/SKR18156592/Fingerprint-Liveness-Detection.git

cd Fingerprint-Liveness-Detection
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Prepare Dataset

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

---

## 4. Train the Model

```bash
python liveness_train.py
```

The trained model will be saved to:

```text
models/liveness_model.keras
```

---

## 5. Evaluate the Model

```bash
python liveness_eval.py
```

Generated outputs will be saved in:

```text
outputs/
```

---

## 6. Run Inference

Update the image path inside:

```python
IMAGE_PATH = "test_images/your_image.jpg"
```

Then run:

```bash
python liveness_infer.py
```

Example output:

```text
Prediction
--------------------
Label      : LIVE ✅
Score      : 0.08
Confidence : 92.14%
```

---

# Results

The evaluation script reports:

* Accuracy
* Precision
* Recall
* F1 Score
* APCER
* BPCER
* ACER
* Equal Error Rate (EER)
* ROC-AUC
* Operating Threshold

---

# Assignment Questions

### Dataset Used

Self-collected fingerprint dataset consisting of **Live** and **Spoof** fingerprint images. The dataset was organized into separate folders and split into **70% training**, **15% validation**, and **15% testing**.

### CNN Backbone

**MobileNetV3-Small** was selected because it is lightweight, computationally efficient, and well suited for mobile and embedded biometric applications while maintaining strong classification performance.

### Threshold Selection

The decision threshold was calibrated using the **validation dataset** by selecting the operating point where **BPCER ≈ 3%**. The corresponding **APCER** was then reported at this threshold.

### Equal Error Rate

The Equal Error Rate (EER) was computed using the ROC curve as the point where **False Positive Rate** and **False Negative Rate** are approximately equal.

### Possible Attack Limitations

The model is expected to perform well against:

* Printed fingerprint attacks
* Screen replay attacks

More sophisticated attacks, such as **3D silicone or latex fingerprint molds**, are likely to be more challenging because they preserve realistic ridge structures and depth information.

### Why Fix BPCER?

BPCER represents inconvenience to genuine users. By fixing **BPCER = 3%**, the system limits the rejection rate of legitimate users while measuring the resulting security level through APCER.

### Future Improvements

Given more time and a larger dataset, future work could include:

* Larger and more diverse datasets
* Advanced spoof attack types
* Cross-device evaluation
* Real-time webcam inference
* Model quantization for mobile deployment
* Vision Transformer or EfficientNet backbones
* Domain adaptation for unseen sensors

---

# Requirements

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

# License

This project was developed for academic purposes as part of a Fingerprint Presentation Attack Detection assignment.
