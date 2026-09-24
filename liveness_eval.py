import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score
from dataset import val_ds, test_ds

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*60)
print("Loading Model and Datasets...")
print("="*60)
model = tf.keras.models.load_model(MODEL_DIR / "liveness_model.keras")

# ==========================================================
# 1. Collect Validation Predictions
# ==========================================================
val_images, val_labels = [], []
for img, lbl in val_ds:
    val_images.append(img)
    val_labels.append(lbl)
val_images = np.concatenate(val_images)
val_labels = np.concatenate(val_labels).flatten()

val_preds = model.predict(val_images).flatten()
# val_preds here represents probability of class 1 (Spoof)

# ==========================================================
# 2. Threshold Calibration (Targeting BPCER ≈ 3%)
# ==========================================================
live_scores = val_preds[val_labels == 0]
spoof_scores = val_preds[val_labels == 1]

thresholds = np.linspace(0.0, 1.0, 101)
best_threshold = 0.5
min_bpcer_diff = float('inf')

bpcer_history = []
apcer_history = []

for t in thresholds:
    bpcer = np.mean(live_scores >= t)  # Live misclassified as Spoof
    apcer = np.mean(spoof_scores < t)   # Spoof misclassified as Live
    
    bpcer_history.append(bpcer)
    apcer_history.append(apcer)
    
    diff = abs(bpcer - 0.03)
    if diff < min_bpcer_diff:
        min_bpcer_diff = diff
        best_threshold = t

if best_threshold >= 0.99 or best_threshold <= 0.01:
    eer_diffs = [abs(a - b) for a, b in zip(apcer_history, bpcer_history)]
    best_threshold = thresholds[np.argmin(eer_diffs)]

print(f"\nCalibrated Threshold (BPCER≈3%): {best_threshold:.4f}")

with open(OUTPUT_DIR / "best_threshold.txt", "w") as f:
    f.write(f"{best_threshold:.6f}\n")

# ==========================================================
# 3. Evaluate Test Dataset
# ==========================================================
test_images, test_labels = [], []
for img, lbl in test_ds:
    test_images.append(img)
    test_labels.append(lbl)
test_images = np.concatenate(test_images)
test_labels = np.concatenate(test_labels).flatten()

test_preds = model.predict(test_images).flatten()
test_preds_binary = (test_preds >= best_threshold).astype(int)

# Compute Confusion Matrix (Rows: Actual [Live, Spoof], Cols: Predicted [Live, Spoof])
cm = confusion_matrix(test_labels, test_preds_binary)
print("\nConfusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

# PAD Metrics
test_live_count = tn + fp
test_spoof_count = fn + tp

bpcer_test = fp / test_live_count if test_live_count > 0 else 0.0
apcer_test = fn / test_spoof_count if test_spoof_count > 0 else 0.0
acer_test = (apcer_test + bpcer_test) / 2.0

accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

# Explicitly compute Precision, Recall, and F1 Score (Method 1)
test_precision = precision_score(test_labels, test_preds_binary, zero_division=0)
test_recall = recall_score(test_labels, test_preds_binary, zero_division=0)
test_f1 = f1_score(test_labels, test_preds_binary, zero_division=0)

print("\nPAD Metrics on Test Set:")
print(f"Precision : {test_precision:.4f}")
print(f"Recall    : {test_recall:.4f}")
print(f"F1 Score  : {test_f1:.4f}")
print(f"APCER     : {apcer_test:.4f}")
print(f"BPCER     : {bpcer_test:.4f}")
print(f"ACER      : {acer_test:.4f}")
print(f"Accuracy  : {accuracy:.4f}")

# Compute ROC and EER for reporting
fpr, tpr, roc_thresholds = roc_curve(test_labels, test_preds)
fnr = 1 - tpr
eer_idx = np.nanargmin(np.absolute(fnr - fpr))
eer_val = (fnr[eer_idx] + fpr[eer_idx]) / 2.0
roc_auc = auc(fpr, tpr)

print(f"EER       : {eer_val:.4f}")
print(f"ROC AUC   : {roc_auc:.4f}")

# Save Metrics to CSV (including F1 Score)
with open(OUTPUT_DIR / "evaluation_metrics.csv", "w") as f:
    f.write("Metric,Value\n")
    f.write(f"Threshold,{best_threshold}\n")
    f.write(f"Precision,{test_precision}\n")
    f.write(f"Recall,{test_recall}\n")
    f.write(f"F1_Score,{test_f1}\n")
    f.write(f"APCER,{apcer_test}\n")
    f.write(f"BPCER,{bpcer_test}\n")
    f.write(f"ACER,{acer_test}\n")
    f.write(f"EER,{eer_val}\n")
    f.write(f"ROC_AUC,{roc_auc}\n")
    f.write(f"Accuracy,{accuracy}\n")

# ==========================================================
# 4. Generate Required Plots
# ==========================================================
# Score Distribution Plot
plt.figure(figsize=(8, 5))
plt.hist(live_scores, bins=20, alpha=0.6, color='green', label='Live Scores')
plt.hist(spoof_scores, bins=20, alpha=0.6, color='red', label='Spoof Scores')
plt.axvline(best_threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold = {best_threshold:.2f}')
plt.title("Score Distribution — Live vs Spoof")
plt.xlabel("Spoof Probability Score")
plt.ylabel("Number of Images")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(OUTPUT_DIR / "score_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# APCER-BPCER Tradeoff Curve
plt.figure(figsize=(8, 5))
plt.plot(thresholds, apcer_history, label='APCER', linewidth=2)
plt.plot(thresholds, bpcer_history, label='BPCER', linewidth=2)
plt.axvline(best_threshold, color='red', linestyle=':', label=f'Operating Threshold ({best_threshold:.2f})')
plt.title("APCER & BPCER across Thresholds")
plt.xlabel("Threshold")
plt.ylabel("Error Rate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(OUTPUT_DIR / "apcer_bpcer_tradeoff.png", dpi=300, bbox_inches='tight')
plt.close()

# ROC Curve Plot
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.scatter(fpr[eer_idx], tpr[eer_idx], color='red', s=80, label=f'EER = {eer_val:.3f}')
plt.title("ROC Curve — Fingerprint Liveness Detection")
plt.xlabel("False Positive Rate (BPCER)")
plt.ylabel("True Positive Rate (1 - APCER)")
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=300, bbox_inches='tight')
plt.close()

print("\nAll Evaluation Outputs Successfully Saved in 'outputs/' folder!")

# import numpy as np
# import tensorflow as tf
# from pathlib import Path
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
# from dataset import val_ds, test_ds

# MODEL_DIR = Path("models")
# OUTPUT_DIR = Path("outputs")
# OUTPUT_DIR.mkdir(exist_ok=True)

# print("="*60)
# print("Loading Model and Datasets...")
# print("="*60)
# model = tf.keras.models.load_model(MODEL_DIR / "liveness_model.keras")

# # Collect Validation True Labels & Predictions
# val_images, val_labels = [], []
# for img, lbl in val_ds:
#     val_images.append(img)
#     val_labels.append(lbl)
# val_images = np.concatenate(val_images)
# val_labels = np.concatenate(val_labels).flatten()

# val_preds = model.predict(val_images).flatten()

# # Threshold Calibration (Balanced EER approach)
# fpr, tpr, thresholds = roc_curve(val_labels, val_preds)
# fnr = 1 - tpr
# eer_idx = np.nanargmin(np.absolute(fnr - fpr))
# best_threshold = thresholds[eer_idx]
# print(f"\nCalibrated EER Threshold: {best_threshold:.4f}")

# # Evaluate Test Dataset
# test_images, test_labels = [], []
# for img, lbl in test_ds:
#     test_images.append(img)
#     test_labels.append(lbl)
# test_images = np.concatenate(test_images)
# test_labels = np.concatenate(test_labels).flatten()

# test_preds = model.predict(test_images).flatten()
# test_preds_binary = (test_preds >= best_threshold).astype(int)

# # Compute Metrics
# cm = confusion_matrix(test_labels, test_preds_binary)
# print("\nConfusion Matrix:")
# print(cm)

# # Save evaluation metric text summary
# with open(OUTPUT_DIR / "evaluation_metrics.csv", "w") as f:
#     f.write(f"Threshold,{best_threshold}\n")
#     f.write(f"ROC_AUC,{auc(fpr, tpr)}\n")

# print("\nEvaluation Completed Successfully!")


# # """
# # Fingerprint Liveness Detection
# # Evaluation Script

# # This script performs:

# # 1. Threshold calibration using VALIDATION set
# # 2. Finds threshold at BPCER = 3%
# # 3. Computes Equal Error Rate (EER)
# # 4. Evaluates TEST set using calibrated threshold
# # 5. Saves all plots and metrics
# # """

# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import tensorflow as tf

# # from pathlib import Path

# # from sklearn.metrics import (
# #     accuracy_score,
# #     precision_score,
# #     recall_score,
# #     f1_score,
# #     confusion_matrix,
# #     classification_report,
# #     roc_curve,
# #     auc
# # )

# # # ==========================================================
# # # Configuration
# # # ==========================================================

# # MODEL_PATH = "models/liveness_model_checkpoint.keras"

# # VAL_DIR = "dataset_split/val"
# # TEST_DIR = "dataset_split/test"

# # OUTPUT_DIR = Path("outputs")
# # OUTPUT_DIR.mkdir(exist_ok=True)

# # IMG_SIZE = (224, 224)
# # BATCH_SIZE = 32

# # # ==========================================================
# # # MobileNetV3 Preprocessing
# # # ==========================================================

# # preprocess = tf.keras.applications.mobilenet_v3.preprocess_input

# # # ==========================================================
# # # Load Validation Dataset
# # # ==========================================================

# # print("=" * 60)
# # print("Loading Validation Dataset...")
# # print("=" * 60)

# # val_ds = tf.keras.utils.image_dataset_from_directory(
# #     VAL_DIR,
# #     image_size=IMG_SIZE,
# #     batch_size=BATCH_SIZE,
# #     label_mode="binary",
# #     shuffle=False
# # )

# # val_ds = val_ds.map(
# #     lambda images, labels: (preprocess(images), labels),
# #     num_parallel_calls=tf.data.AUTOTUNE
# # ).prefetch(tf.data.AUTOTUNE)

# # # ==========================================================
# # # Load Test Dataset
# # # ==========================================================

# # print("\nLoading Test Dataset...")

# # test_ds = tf.keras.utils.image_dataset_from_directory(
# #     TEST_DIR,
# #     image_size=IMG_SIZE,
# #     batch_size=BATCH_SIZE,
# #     label_mode="binary",
# #     shuffle=False
# # )

# # test_ds = test_ds.map(
# #     lambda images, labels: (preprocess(images), labels),
# #     num_parallel_calls=tf.data.AUTOTUNE
# # ).prefetch(tf.data.AUTOTUNE)

# # # ==========================================================
# # # Load Model
# # # ==========================================================

# # print("\nLoading trained model...")

# # model = tf.keras.models.load_model(MODEL_PATH)

# # print("Model loaded successfully.")

# # # ==========================================================
# # # Predict Validation Scores
# # # ==========================================================

# # print("\nPredicting Validation Scores...")

# # val_scores = model.predict(val_ds).flatten()

# # val_labels = []

# # for _, labels in val_ds:
# #     val_labels.extend(labels.numpy())

# # val_labels = np.array(val_labels).astype(int).flatten()

# # print(f"Validation Images : {len(val_scores)}")

# # # ==========================================================
# # # Threshold Calibration on Validation Set
# # # ==========================================================

# # print("\n" + "=" * 60)
# # print("Threshold Calibration")
# # print("=" * 60)

# # threshold_values = np.arange(0.00, 1.01, 0.01)

# # apcer_values = []
# # bpcer_values = []
# # acer_values = []

# # for threshold in threshold_values:

# #     # Probability >= threshold -> Spoof (1)
# #     predictions = (val_scores >= threshold).astype(int)

# #     cm = confusion_matrix(val_labels, predictions)

# #     # Handle edge cases
# #     if cm.shape != (2, 2):
# #         apcer = 0.0
# #         bpcer = 0.0
# #         acer = 0.0
# #     else:
# #         tn, fp, fn, tp = cm.ravel()

# #         # APCER = Spoof classified as Live
# #         apcer = fn / (tp + fn) if (tp + fn) > 0 else 0

# #         # BPCER = Live classified as Spoof
# #         bpcer = fp / (tn + fp) if (tn + fp) > 0 else 0

# #         acer = (apcer + bpcer) / 2

# #     apcer_values.append(apcer)
# #     bpcer_values.append(bpcer)
# #     acer_values.append(acer)

# # # ==========================================================
# # # Find Threshold at BPCER = 3%
# # # ==========================================================

# # TARGET_BPCER = 0.03

# # best_index = np.argmin(
# #     np.abs(np.array(bpcer_values) - TARGET_BPCER)
# # )

# # best_threshold = threshold_values[best_index]

# # best_bpcer = bpcer_values[best_index]
# # best_apcer = apcer_values[best_index]
# # best_acer = acer_values[best_index]

# # print("\nOperating Threshold")
# # print("-" * 40)
# # print(f"Target BPCER : {TARGET_BPCER:.2%}")
# # print(f"Threshold    : {best_threshold:.2f}")
# # print(f"BPCER        : {best_bpcer:.4f}")
# # print(f"APCER        : {best_apcer:.4f}")
# # print(f"ACER         : {best_acer:.4f}")

# # # ==========================================================
# # # Equal Error Rate (EER)
# # # ==========================================================

# # fpr, tpr, roc_thresholds = roc_curve(val_labels, val_scores)

# # fnr = 1 - tpr

# # eer_index = np.nanargmin(np.abs(fpr - fnr))

# # eer = (fpr[eer_index] + fnr[eer_index]) / 2

# # eer_threshold = roc_thresholds[eer_index]

# # roc_auc = auc(fpr, tpr)

# # print("\nEqual Error Rate")
# # print("-" * 40)
# # print(f"EER          : {eer:.4f}")
# # print(f"EER Threshold: {eer_threshold:.4f}")
# # print(f"ROC AUC      : {roc_auc:.4f}")


# # # ==========================================================
# # # Evaluate Test Set using Calibrated Threshold
# # # ==========================================================

# # print("\n" + "=" * 60)
# # print("Evaluating Test Dataset")
# # print("=" * 60)

# # # Predict probabilities on test dataset
# # test_scores = model.predict(test_ds).flatten()

# # # Collect true labels
# # test_labels = []

# # for _, labels in test_ds:
# #     test_labels.extend(labels.numpy())

# # test_labels = np.array(test_labels).astype(int).flatten()

# # # Apply calibrated threshold
# # test_predictions = (test_scores >= best_threshold).astype(int)

# # # ==========================================================
# # # Evaluation Metrics
# # # ==========================================================

# # accuracy = accuracy_score(test_labels, test_predictions)

# # precision = precision_score(
# #     test_labels,
# #     test_predictions,
# #     zero_division=0
# # )

# # recall = recall_score(
# #     test_labels,
# #     test_predictions,
# #     zero_division=0
# # )

# # f1 = f1_score(
# #     test_labels,
# #     test_predictions,
# #     zero_division=0
# # )

# # print("\nClassification Metrics")
# # print("-" * 40)

# # print(f"Accuracy  : {accuracy:.4f}")
# # print(f"Precision : {precision:.4f}")
# # print(f"Recall    : {recall:.4f}")
# # print(f"F1 Score  : {f1:.4f}")

# # # ==========================================================
# # # Confusion Matrix
# # # ==========================================================

# # cm = confusion_matrix(test_labels, test_predictions)

# # print("\nConfusion Matrix")
# # print("-" * 40)

# # print(cm)

# # tn, fp, fn, tp = cm.ravel()

# # print("\nDetailed Counts")
# # print("-" * 40)

# # print(f"True Negatives : {tn}")
# # print(f"False Positives: {fp}")
# # print(f"False Negatives: {fn}")
# # print(f"True Positives : {tp}")

# # # ==========================================================
# # # APCER / BPCER / ACER (Test Set)
# # # ==========================================================

# # # APCER = Spoof incorrectly classified as Live
# # apcer = fn / (tp + fn) if (tp + fn) > 0 else 0

# # # BPCER = Live incorrectly classified as Spoof
# # bpcer = fp / (tn + fp) if (tn + fp) > 0 else 0

# # acer = (apcer + bpcer) / 2

# # print("\nPAD Metrics")
# # print("-" * 40)

# # print(f"APCER : {apcer:.4f}")
# # print(f"BPCER : {bpcer:.4f}")
# # print(f"ACER  : {acer:.4f}")

# # print("\nClassification Report")
# # print("-" * 40)

# # print(
# #     classification_report(
# #         test_labels,
# #         test_predictions,
# #         target_names=["Live", "Spoof"],
# #         zero_division=0
# #     )
# # )

# # # ==========================================================
# # # Save Confusion Matrix Figure
# # # ==========================================================

# # plt.figure(figsize=(6,6))

# # plt.imshow(cm, cmap="Blues")

# # plt.title("Confusion Matrix")

# # plt.colorbar()

# # classes = ["Live", "Spoof"]

# # ticks = np.arange(len(classes))

# # plt.xticks(ticks, classes)
# # plt.yticks(ticks, classes)

# # for i in range(cm.shape[0]):
# #     for j in range(cm.shape[1]):
# #         plt.text(
# #             j,
# #             i,
# #             str(cm[i, j]),
# #             ha="center",
# #             va="center",
# #             fontsize=12
# #         )

# # plt.xlabel("Predicted")
# # plt.ylabel("Actual")

# # plt.tight_layout()

# # plt.savefig(
# #     OUTPUT_DIR / "confusion_matrix.png",
# #     dpi=300
# # )

# # plt.close()

# # print("\nConfusion matrix saved.")

# # # ==========================================================
# # # Score Distribution
# # # ==========================================================

# # live_scores = test_scores[test_labels == 0]
# # spoof_scores = test_scores[test_labels == 1]

# # plt.figure(figsize=(8,5))

# # plt.hist(
# #     live_scores,
# #     bins=20,
# #     alpha=0.6,
# #     color="green",
# #     label="Live"
# # )

# # plt.hist(
# #     spoof_scores,
# #     bins=20,
# #     alpha=0.6,
# #     color="red",
# #     label="Spoof"
# # )

# # plt.axvline(
# #     best_threshold,
# #     color="black",
# #     linestyle="--",
# #     linewidth=2,
# #     label=f"Threshold = {best_threshold:.2f}"
# # )

# # plt.xlabel("Spoof Probability")
# # plt.ylabel("Number of Images")
# # plt.title("Score Distribution")

# # plt.legend()

# # plt.tight_layout()

# # plt.savefig(
# #     OUTPUT_DIR / "score_distribution.png",
# #     dpi=300
# # )

# # plt.close()

# # print("✓ Score distribution saved.")

# # # ==========================================================
# # # APCER vs BPCER Tradeoff Curve
# # # ==========================================================

# # plt.figure(figsize=(6,6))

# # plt.plot(
# #     bpcer_values,
# #     apcer_values,
# #     linewidth=2,
# #     label="Tradeoff Curve"
# # )

# # plt.scatter(
# #     best_bpcer,
# #     best_apcer,
# #     color="red",
# #     s=80,
# #     label=f"BPCER=3% Threshold ({best_threshold:.2f})"
# # )

# # plt.xlabel("BPCER")
# # plt.ylabel("APCER")
# # plt.title("APCER-BPCER Tradeoff Curve")

# # plt.grid(True)
# # plt.legend()

# # plt.tight_layout()

# # plt.savefig(
# #     OUTPUT_DIR / "apcer_bpcer_tradeoff.png",
# #     dpi=300
# # )

# # plt.close()

# # print("✓ APCER-BPCER tradeoff curve saved.")

# # # ==========================================================
# # # ROC Curve
# # # ==========================================================

# # plt.figure(figsize=(6,6))

# # plt.plot(
# #     fpr,
# #     tpr,
# #     linewidth=2,
# #     label=f"AUC = {roc_auc:.3f}"
# # )

# # plt.plot(
# #     [0,1],
# #     [0,1],
# #     linestyle="--",
# #     color="gray"
# # )

# # plt.scatter(
# #     fpr[eer_index],
# #     tpr[eer_index],
# #     color="red",
# #     s=80,
# #     label=f"EER = {eer:.3f}"
# # )

# # plt.xlabel("False Positive Rate")
# # plt.ylabel("True Positive Rate")

# # plt.title("ROC Curve")

# # plt.legend()

# # plt.tight_layout()

# # plt.savefig(
# #     OUTPUT_DIR / "roc_curve.png",
# #     dpi=300
# # )

# # plt.close()

# # print("✓ ROC curve saved.")

# # # ==========================================================
# # # Save Best Threshold
# # # ==========================================================

# # threshold_file = OUTPUT_DIR / "best_threshold.txt"

# # with open(threshold_file, "w") as f:
# #     f.write(f"{best_threshold:.6f}")

# # print("✓ Best threshold saved.")

# # # ==========================================================
# # # Save Evaluation Metrics
# # # ==========================================================

# # results = pd.DataFrame({

# #     "Metric":[
# #         "Accuracy",
# #         "Precision",
# #         "Recall",
# #         "F1 Score",
# #         "APCER",
# #         "BPCER",
# #         "ACER",
# #         "ROC AUC",
# #         "EER",
# #         "Threshold (BPCER≈3%)"
# #     ],

# #     "Value":[
# #         accuracy,
# #         precision,
# #         recall,
# #         f1,
# #         apcer,
# #         bpcer,
# #         acer,
# #         roc_auc,
# #         eer,
# #         best_threshold
# #     ]
# # })

# # results.to_csv(
# #     OUTPUT_DIR / "evaluation_metrics.csv",
# #     index=False
# # )

# # print("✓ Evaluation metrics saved.")

# # # ==========================================================
# # # Final Summary
# # # ==========================================================

# # print("\n" + "="*60)
# # print("FINAL RESULTS")
# # print("="*60)

# # print(f"Threshold (BPCER≈3%) : {best_threshold:.2f}")
# # print(f"APCER               : {best_apcer:.4f}")
# # print(f"BPCER               : {best_bpcer:.4f}")
# # print(f"ACER                : {best_acer:.4f}")
# # print(f"EER                 : {eer:.4f}")
# # print(f"ROC AUC             : {roc_auc:.4f}")

# # print("\nTest Set Performance")
# # print("-"*40)

# # print(f"Accuracy  : {accuracy:.4f}")
# # print(f"Precision : {precision:.4f}")
# # print(f"Recall    : {recall:.4f}")
# # print(f"F1 Score  : {f1:.4f}")

# # print("\nAll outputs saved in:")
# # print(OUTPUT_DIR)

# # print("\nEvaluation Completed Successfully!")
