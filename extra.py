# """
# Fingerprint Liveness Detection
# Evaluation Script

# This script:
# 1. Loads the trained TensorFlow model.
# 2. Loads the test dataset.
# 3. Preprocesses images using MobileNetV3 preprocessing.
# 4. Predicts liveness scores.
# 5. Converts probabilities into class labels.
# 6. Computes evaluation metrics.
# 7. Displays the confusion matrix.

# Class Labels:
# 0 -> Live Fingerprint
# 1 -> Spoof Fingerprint
# """

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import tensorflow as tf

# from pathlib import Path

# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     confusion_matrix,
#     classification_report,
#     roc_curve,
#     auc,
# )

# # ==========================================================
# # Configuration
# # ==========================================================

# # Path to trained model
# MODEL_PATH = "models/liveness_model.keras"

# # Test dataset directory
# TEST_DIR = "dataset_split/test"

# # Output directory (used later for saving plots)
# OUTPUT_DIR = Path("outputs")
# OUTPUT_DIR.mkdir(exist_ok=True)

# # Image size used during training
# IMG_SIZE = (224, 224)

# # Batch size
# BATCH_SIZE = 32


# # ==========================================================
# # Load Test Dataset
# # ==========================================================

# print("=" * 60)
# print("Loading test dataset...")
# print("=" * 60)

# test_ds = tf.keras.utils.image_dataset_from_directory(
#     TEST_DIR,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     label_mode="binary",
#     shuffle=False      # IMPORTANT:
#                        # Keep order fixed so predictions and labels match.
# )

# print("\nClass Mapping:")
# print(test_ds.class_names)


# # ==========================================================
# # Apply Same Preprocessing Used During Training
# # ==========================================================

# # MobileNetV3 expects images to be preprocessed
# preprocess = tf.keras.applications.mobilenet_v3.preprocess_input

# test_ds = test_ds.map(
#     lambda images, labels: (preprocess(images), labels),
#     num_parallel_calls=tf.data.AUTOTUNE
# )

# # Prefetch batches for faster inference
# test_ds = test_ds.prefetch(tf.data.AUTOTUNE)


# # ==========================================================
# # Load Trained Model
# # ==========================================================

# print("\nLoading trained model...")

# model = tf.keras.models.load_model(MODEL_PATH)

# print("Model loaded successfully.")


# # ==========================================================
# # Predict Liveness Scores
# # ==========================================================

# print("\nGenerating predictions...")

# # Predict probability for every image
# scores = model.predict(test_ds)

# # Convert shape from (N,1) to (N,)
# scores = scores.flatten()

# print(f"Total predictions: {len(scores)}")


# # ==========================================================
# # Collect Ground Truth Labels
# # ==========================================================

# true_labels = []

# # Iterate through dataset and collect labels
# for _, labels in test_ds:
#     true_labels.extend(labels.numpy())

# # Convert list into NumPy array
# true_labels = np.array(true_labels).astype(int).flatten()

# print(f"Total labels: {len(true_labels)}")


# # ==========================================================
# # Convert Probabilities into Class Predictions
# # ==========================================================

# # Initial decision threshold
# threshold = 0.5

# # Probability >= threshold -> Spoof (1)
# # Probability < threshold  -> Live (0)
# predictions = (scores >= threshold).astype(int)


# # ==========================================================
# # Compute Evaluation Metrics
# # ==========================================================

# accuracy = accuracy_score(
#     true_labels,
#     predictions
# )

# precision = precision_score(
#     true_labels,
#     predictions
# )

# recall = recall_score(
#     true_labels,
#     predictions
# )

# f1 = f1_score(
#     true_labels,
#     predictions
# )

# print("\nEvaluation Metrics")
# print("-" * 30)

# print(f"Accuracy : {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall   : {recall:.4f}")
# print(f"F1 Score : {f1:.4f}")


# # ==========================================================
# # Classification Report
# # ==========================================================

# print("\nClassification Report")
# print("-" * 30)

# print(
#     classification_report(
#         true_labels,
#         predictions,
#         target_names=["Live", "Spoof"]
#     )
# )


# # ==========================================================
# # Confusion Matrix
# # ==========================================================

# cm = confusion_matrix(
#     true_labels,
#     predictions
# )

# print("\nConfusion Matrix")
# print("-" * 30)

# print(cm)

# # Matrix Layout
# #
# #                Predicted
# #               Live   Spoof
# #
# # Actual Live    TN      FP
# #
# # Actual Spoof   FN      TP

# tn, fp, fn, tp = cm.ravel()

# print("\nDetailed Counts")
# print("-" * 30)

# print(f"True Negatives : {tn}")
# print(f"False Positives: {fp}")
# print(f"False Negatives: {fn}")
# print(f"True Positives : {tp}")

# # ==========================================================
# # APCER / BPCER / ACER
# # ==========================================================



# # Attack Presentation Classification Error Rate
# # (Spoof classified as Live)
# apcer = fn / (tp + fn) if (tp + fn) > 0 else 0

# # Bona Fide Presentation Classification Error Rate
# # (Live classified as Spoof)
# bpcer = fp / (tn + fp) if (tn + fp) > 0 else 0

# # Average Classification Error Rate
# acer = (apcer + bpcer) / 2

# print("\nBiometric Metrics")
# print("-" * 30)
# print(f"APCER : {apcer:.4f}")
# print(f"BPCER : {bpcer:.4f}")
# print(f"ACER  : {acer:.4f}")


# # ==========================================================
# # Save Confusion Matrix Figure
# # ==========================================================

# plt.figure(figsize=(5, 5))

# plt.imshow(cm, interpolation="nearest", cmap="Blues")

# plt.title("Confusion Matrix")

# plt.colorbar()

# classes = ["Live", "Spoof"]

# ticks = np.arange(len(classes))

# plt.xticks(ticks, classes)
# plt.yticks(ticks, classes)

# for i in range(cm.shape[0]):
#     for j in range(cm.shape[1]):
#         plt.text(
#             j,
#             i,
#             str(cm[i, j]),
#             ha="center",
#             va="center",
#             fontsize=12
#         )

# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")

# plt.tight_layout()

# plt.savefig(
#     OUTPUT_DIR / "confusion_matrix.png",
#     dpi=300
# )

# plt.close()


# # ==========================================================
# # ROC Curve
# # ==========================================================

# fpr, tpr, roc_thresholds = roc_curve(
#     true_labels,
#     scores
# )

# roc_auc = auc(fpr, tpr)

# print(f"\nROC AUC : {roc_auc:.4f}")

# # ==========================================================
# # Equal Error Rate (EER)
# # ==========================================================

# # False Negative Rate (same as BPCER curve on ROC)
# fnr = 1 - tpr

# # Find operating point where FPR ~= FNR
# eer_index = np.nanargmin(np.abs(fpr - fnr))

# eer = (fpr[eer_index] + fnr[eer_index]) / 2
# eer_threshold = roc_thresholds[eer_index]

# print("\nEqual Error Rate")
# print("-" * 30)
# print(f"EER       : {eer:.4f}")
# print(f"Threshold : {eer_threshold:.4f}")

# plt.figure(figsize=(6, 6))

# plt.plot(
#     fpr,
#     tpr,
#     linewidth=2,
#     label=f"AUC = {roc_auc:.3f}"
# )

# plt.plot(
#     [0, 1],
#     [0, 1],
#     linestyle="--"
# )

# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend()

# plt.tight_layout()

# plt.savefig(
#     OUTPUT_DIR / "roc_curve.png",
#     dpi=300
# )

# plt.close()


# # ==========================================================
# # Score Distribution
# # ==========================================================

# live_scores = scores[true_labels == 0]
# spoof_scores = scores[true_labels == 1]

# plt.figure(figsize=(8, 5))

# plt.hist(
#     live_scores,
#     bins=20,
#     alpha=0.6,
#     label="Live"
# )

# plt.hist(
#     spoof_scores,
#     bins=20,
#     alpha=0.6,
#     label="Spoof"
# )

# plt.axvline(
#     threshold,
#     color="red",
#     linestyle="--",
#     linewidth=2,
#     label=f"Threshold = {threshold}"
# )

# plt.xlabel("Spoof Probability")
# plt.ylabel("Number of Images")
# plt.title("Score Distribution")
# plt.legend()

# plt.tight_layout()

# plt.savefig(
#     OUTPUT_DIR / "score_distribution.png",
#     dpi=300
# )

# plt.close()


# # ==========================================================
# # Threshold Sweep
# # ==========================================================

# threshold_values = np.linspace(0, 1, 101)

# apcer_values = []
# bpcer_values = []
# acer_values = []

# for t in threshold_values:

#     pred = (scores >= t).astype(int)

#     cm_temp = confusion_matrix(
#         true_labels,
#         pred
#     )

#     tn, fp, fn, tp = cm_temp.ravel()

#     apcer_temp = fn / (tp + fn) if (tp + fn) > 0 else 0

#     bpcer_temp = fp / (tn + fp) if (tn + fp) > 0 else 0

#     acer_temp = (apcer_temp + bpcer_temp) / 2

#     apcer_values.append(apcer_temp)
#     bpcer_values.append(bpcer_temp)
#     acer_values.append(acer_temp)


# # ==========================================================
# # APCER / BPCER Curve
# # ==========================================================

# plt.figure(figsize=(8, 5))

# plt.plot(
#     threshold_values,
#     apcer_values,
#     linewidth=2,
#     label="APCER"
# )

# plt.plot(
#     threshold_values,
#     bpcer_values,
#     linewidth=2,
#     label="BPCER"
# )

# plt.xlabel("Threshold")
# plt.ylabel("Error Rate")
# plt.title("APCER vs BPCER")
# plt.grid(True)
# plt.legend()

# plt.tight_layout()

# plt.savefig(
#     OUTPUT_DIR / "apcer_bpcer_curve.png",
#     dpi=300
# )

# plt.close()


# # ==========================================================
# # Best Threshold
# # ==========================================================

# best_index = np.argmin(acer_values)

# best_threshold = threshold_values[best_index]

# print("\nRe-evaluating using best threshold...")

# predictions = (scores >= best_threshold).astype(int)

# cm = confusion_matrix(true_labels, predictions)

# accuracy = accuracy_score(true_labels, predictions)
# precision = precision_score(true_labels, predictions)
# recall = recall_score(true_labels, predictions)
# f1 = f1_score(true_labels, predictions)


# best_acer = acer_values[best_index]

# print("\nOptimal Threshold")
# print("-" * 30)
# print(f"Threshold : {best_threshold:.2f}")
# print(f"Minimum ACER : {best_acer:.4f}")


# # Save threshold for inference
# threshold_file = OUTPUT_DIR / "best_threshold.txt"

# with open(threshold_file, "w") as f:
#     f.write(f"{best_threshold:.6f}")

# print(f"Best threshold saved to: {threshold_file}")


# # ==========================================================
# # Save Evaluation Metrics
# # ==========================================================

# results = pd.DataFrame({
#     "Metric": [
#         "Accuracy",
#         "Precision",
#         "Recall",
#         "F1 Score",
#         "APCER",
#         "BPCER",
#         "ACER",
#         "ROC AUC",
#         "Best Threshold"
#     ],
#     "Value": [
#         accuracy,
#         precision,
#         recall,
#         f1,
#         apcer,
#         bpcer,
#         acer,
#         roc_auc,
#         best_threshold
#     ]
# })

# results.to_csv(
#     OUTPUT_DIR / "evaluation_metrics.csv",
#     index=False
# )

# print("\nEvaluation metrics saved to:")
# print(OUTPUT_DIR / "evaluation_metrics.csv")

# print("\nEvaluation completed successfully.")