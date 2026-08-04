# Fingerprint Liveness Detection using TensorFlow

## Overview

This project implements a fingerprint liveness detection system using deep learning to distinguish between genuine (live) and spoof fingerprints. The model is built using TensorFlow and MobileNetV3-Small with transfer learning.

The project includes:

- Data preprocessing
- Model training
- Model evaluation
- Threshold calibration
- Single image inference

---

## Project Structure

```text
Fingerprint-Liveness-Detection/
│
├── data/
│   ├── raw/
│   ├── live/
│   └── spoof/
│
├── dataset_split/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   └── liveness_model.keras
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── score_distribution.png
│   ├── apcer_bpcer_curve.png
│   ├── evaluation_metrics.csv
│   └── best_threshold.txt
│
├── test_images/
│
├── liveness_train.py
├── liveness_eval.py
├── liveness_infer.py
├── requirements.txt
├── report.pdf
└── README.md
```

---

## Model

- TensorFlow
- MobileNetV3-Small
- Transfer Learning
- Binary Classification

Classes

- Live Fingerprint
- Spoof Fingerprint

---

## Training

```bash
python liveness_train.py
```

The trained model is saved to

```
models/liveness_model.keras
```

---

## Evaluation

```bash
python liveness_eval.py
```

Generated outputs

- Confusion Matrix
- ROC Curve
- Score Distribution
- APCER/BPCER Curve
- Evaluation Metrics CSV

---

## Inference

```bash
python liveness_infer.py
```

Predicts whether a fingerprint is

- Live
- Spoof

along with the confidence score.

---

## Evaluation Metrics

The project reports

- Accuracy
- Precision
- Recall
- F1-score
- APCER
- BPCER
- ACER
- ROC-AUC

---

## Future Improvements

- Larger dataset
- Data augmentation
- Fine-tuning MobileNetV3
- Real-time webcam inference
- Fingerprint quality assessment

---

## Author

Suman Raj

IIT Kharagpur