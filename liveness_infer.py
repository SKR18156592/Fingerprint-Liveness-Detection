"""
Fingerprint Liveness Detection
Inference Script

Predict whether a fingerprint image is:

- Live
- Spoof
"""

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ==========================================================
# Configuration
# ==========================================================

MODEL_PATH = "models/liveness_model.keras"

THRESHOLD_PATH = "outputs/best_threshold.txt"

IMG_SIZE = (224, 224)

IMAGE_PATH = "test_images/1__M_Left_index_finger.BMP"     # Change this image


# ==========================================================
# Load Model
# ==========================================================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded.")


# ==========================================================
# Load Threshold
# ==========================================================

with open(THRESHOLD_PATH, "r") as f:
    threshold = float(f.read())

print(f"Threshold: {threshold:.4f}")


# ==========================================================
# Load Image
# ==========================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Cannot read image: {IMAGE_PATH}"
    )

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# ==========================================================
# Preprocess (same as training)
# ==========================================================

input_image = cv2.resize(
    image_rgb,
    IMG_SIZE
)

input_image = input_image.astype(np.float32)

# Normalize to [0,1]
input_image = input_image / 255.0

# Same ImageNet normalization used during training
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

input_image = (input_image - IMAGENET_MEAN) / IMAGENET_STD

input_image = np.expand_dims(
    input_image,
    axis=0
)

# ==========================================================
# Predict
# ==========================================================

score = model.predict(
    input_image,
    verbose=0
)[0][0]

prediction = int(score >= threshold)

confidence = score if prediction == 1 else 1 - score

label = "SPOOF ❌" if prediction == 1 else "LIVE ✅"


print("\nPrediction")
print("------------------------")
print(f"Label      : {label}")
print(f"Score      : {score:.4f}")
print(f"Confidence : {confidence:.2%}")


# ==========================================================
# Display Image
# ==========================================================

plt.figure(figsize=(5, 5))

plt.imshow(image_rgb)

plt.axis("off")

plt.title(
    f"{label}\n"
    f"Spoof Probability: {score:.3f}\n"
    f"Confidence: {confidence:.2%}"
)
plt.show()