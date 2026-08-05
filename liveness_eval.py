"""
Fingerprint Liveness Detection
Evaluation Script

This script performs:

1. Threshold calibration using VALIDATION set
2. Finds threshold at BPCER = 3%
3. Computes Equal Error Rate (EER)
4. Evaluates TEST set using calibrated threshold
5. Saves all plots and metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# ==========================================================
# Configuration
# ==========================================================

MODEL_PATH = "models/liveness_model_checkpoint.keras"

VAL_DIR = "dataset_split/val"
TEST_DIR = "dataset_split/test"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==========================================================
# MobileNetV3 Preprocessing
# ==========================================================

preprocess = tf.keras.applications.mobilenet_v3.preprocess_input

# ==========================================================
# Load Validation Dataset
# ==========================================================

print("=" * 60)
print("Loading Validation Dataset...")
print("=" * 60)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

val_ds = val_ds.map(
    lambda images, labels: (preprocess(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# ==========================================================
# Load Test Dataset
# ==========================================================

print("\nLoading Test Dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

test_ds = test_ds.map(
    lambda images, labels: (preprocess(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# ==========================================================
# Load Model
# ==========================================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ==========================================================
# Predict Validation Scores
# ==========================================================

print("\nPredicting Validation Scores...")

val_scores = model.predict(val_ds).flatten()

val_labels = []

for _, labels in val_ds:
    val_labels.extend(labels.numpy())

val_labels = np.array(val_labels).astype(int).flatten()

print(f"Validation Images : {len(val_scores)}")

# ==========================================================
# Threshold Calibration on Validation Set
# ==========================================================

print("\n" + "=" * 60)
print("Threshold Calibration")
print("=" * 60)

threshold_values = np.arange(0.00, 1.01, 0.01)

apcer_values = []
bpcer_values = []
acer_values = []

for threshold in threshold_values:

    # Probability >= threshold -> Spoof (1)
    predictions = (val_scores >= threshold).astype(int)

    cm = confusion_matrix(val_labels, predictions)

    # Handle edge cases
    if cm.shape != (2, 2):
        apcer = 0.0
        bpcer = 0.0
        acer = 0.0
    else:
        tn, fp, fn, tp = cm.ravel()

        # APCER = Spoof classified as Live
        apcer = fn / (tp + fn) if (tp + fn) > 0 else 0

        # BPCER = Live classified as Spoof
        bpcer = fp / (tn + fp) if (tn + fp) > 0 else 0

        acer = (apcer + bpcer) / 2

    apcer_values.append(apcer)
    bpcer_values.append(bpcer)
    acer_values.append(acer)

# ==========================================================
# Find Threshold at BPCER = 3%
# ==========================================================

TARGET_BPCER = 0.03

best_index = np.argmin(
    np.abs(np.array(bpcer_values) - TARGET_BPCER)
)

best_threshold = threshold_values[best_index]

best_bpcer = bpcer_values[best_index]
best_apcer = apcer_values[best_index]
best_acer = acer_values[best_index]

print("\nOperating Threshold")
print("-" * 40)
print(f"Target BPCER : {TARGET_BPCER:.2%}")
print(f"Threshold    : {best_threshold:.2f}")
print(f"BPCER        : {best_bpcer:.4f}")
print(f"APCER        : {best_apcer:.4f}")
print(f"ACER         : {best_acer:.4f}")

# ==========================================================
# Equal Error Rate (EER)
# ==========================================================

fpr, tpr, roc_thresholds = roc_curve(val_labels, val_scores)

fnr = 1 - tpr

eer_index = np.nanargmin(np.abs(fpr - fnr))

eer = (fpr[eer_index] + fnr[eer_index]) / 2

eer_threshold = roc_thresholds[eer_index]

roc_auc = auc(fpr, tpr)

print("\nEqual Error Rate")
print("-" * 40)
print(f"EER          : {eer:.4f}")
print(f"EER Threshold: {eer_threshold:.4f}")
print(f"ROC AUC      : {roc_auc:.4f}")


# ==========================================================
# Evaluate Test Set using Calibrated Threshold
# ==========================================================

print("\n" + "=" * 60)
print("Evaluating Test Dataset")
print("=" * 60)

# Predict probabilities on test dataset
test_scores = model.predict(test_ds).flatten()

# Collect true labels
test_labels = []

for _, labels in test_ds:
    test_labels.extend(labels.numpy())

test_labels = np.array(test_labels).astype(int).flatten()

# Apply calibrated threshold
test_predictions = (test_scores >= best_threshold).astype(int)

# ==========================================================
# Evaluation Metrics
# ==========================================================

accuracy = accuracy_score(test_labels, test_predictions)

precision = precision_score(
    test_labels,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    test_labels,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    test_labels,
    test_predictions,
    zero_division=0
)

print("\nClassification Metrics")
print("-" * 40)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(test_labels, test_predictions)

print("\nConfusion Matrix")
print("-" * 40)

print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nDetailed Counts")
print("-" * 40)

print(f"True Negatives : {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives : {tp}")

# ==========================================================
# APCER / BPCER / ACER (Test Set)
# ==========================================================

# APCER = Spoof incorrectly classified as Live
apcer = fn / (tp + fn) if (tp + fn) > 0 else 0

# BPCER = Live incorrectly classified as Spoof
bpcer = fp / (tn + fp) if (tn + fp) > 0 else 0

acer = (apcer + bpcer) / 2

print("\nPAD Metrics")
print("-" * 40)

print(f"APCER : {apcer:.4f}")
print(f"BPCER : {bpcer:.4f}")
print(f"ACER  : {acer:.4f}")

print("\nClassification Report")
print("-" * 40)

print(
    classification_report(
        test_labels,
        test_predictions,
        target_names=["Live", "Spoof"],
        zero_division=0
    )
)

# ==========================================================
# Save Confusion Matrix Figure
# ==========================================================

plt.figure(figsize=(6,6))

plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix")

plt.colorbar()

classes = ["Live", "Spoof"]

ticks = np.arange(len(classes))

plt.xticks(ticks, classes)
plt.yticks(ticks, classes)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=12
        )

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()

print("\nConfusion matrix saved.")

# ==========================================================
# Score Distribution
# ==========================================================

live_scores = test_scores[test_labels == 0]
spoof_scores = test_scores[test_labels == 1]

plt.figure(figsize=(8,5))

plt.hist(
    live_scores,
    bins=20,
    alpha=0.6,
    color="green",
    label="Live"
)

plt.hist(
    spoof_scores,
    bins=20,
    alpha=0.6,
    color="red",
    label="Spoof"
)

plt.axvline(
    best_threshold,
    color="black",
    linestyle="--",
    linewidth=2,
    label=f"Threshold = {best_threshold:.2f}"
)

plt.xlabel("Spoof Probability")
plt.ylabel("Number of Images")
plt.title("Score Distribution")

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "score_distribution.png",
    dpi=300
)

plt.close()

print("✓ Score distribution saved.")

# ==========================================================
# APCER vs BPCER Tradeoff Curve
# ==========================================================

plt.figure(figsize=(6,6))

plt.plot(
    bpcer_values,
    apcer_values,
    linewidth=2,
    label="Tradeoff Curve"
)

plt.scatter(
    best_bpcer,
    best_apcer,
    color="red",
    s=80,
    label=f"BPCER=3% Threshold ({best_threshold:.2f})"
)

plt.xlabel("BPCER")
plt.ylabel("APCER")
plt.title("APCER-BPCER Tradeoff Curve")

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "apcer_bpcer_tradeoff.png",
    dpi=300
)

plt.close()

print("✓ APCER-BPCER tradeoff curve saved.")

# ==========================================================
# ROC Curve
# ==========================================================

plt.figure(figsize=(6,6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {roc_auc:.3f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--",
    color="gray"
)

plt.scatter(
    fpr[eer_index],
    tpr[eer_index],
    color="red",
    s=80,
    label=f"EER = {eer:.3f}"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "roc_curve.png",
    dpi=300
)

plt.close()

print("✓ ROC curve saved.")

# ==========================================================
# Save Best Threshold
# ==========================================================

threshold_file = OUTPUT_DIR / "best_threshold.txt"

with open(threshold_file, "w") as f:
    f.write(f"{best_threshold:.6f}")

print("✓ Best threshold saved.")

# ==========================================================
# Save Evaluation Metrics
# ==========================================================

results = pd.DataFrame({

    "Metric":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "APCER",
        "BPCER",
        "ACER",
        "ROC AUC",
        "EER",
        "Threshold (BPCER≈3%)"
    ],

    "Value":[
        accuracy,
        precision,
        recall,
        f1,
        apcer,
        bpcer,
        acer,
        roc_auc,
        eer,
        best_threshold
    ]
})

results.to_csv(
    OUTPUT_DIR / "evaluation_metrics.csv",
    index=False
)

print("✓ Evaluation metrics saved.")

# ==========================================================
# Final Summary
# ==========================================================

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

print(f"Threshold (BPCER≈3%) : {best_threshold:.2f}")
print(f"APCER               : {best_apcer:.4f}")
print(f"BPCER               : {best_bpcer:.4f}")
print(f"ACER                : {best_acer:.4f}")
print(f"EER                 : {eer:.4f}")
print(f"ROC AUC             : {roc_auc:.4f}")

print("\nTest Set Performance")
print("-"*40)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nAll outputs saved in:")
print(OUTPUT_DIR)

print("\nEvaluation Completed Successfully!")
