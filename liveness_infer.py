import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path

MODEL_DIR = Path("models")
model_path = MODEL_DIR / "liveness_model.keras"

print("Loading model...")
model = tf.keras.models.load_model(model_path)
print("Model loaded.")

THRESHOLD = 0.5  # Or load dynamically from your evaluation file

def predict_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error loading image: {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    
    score = float(model.predict(img)[0][0])
    
    # Assuming Class 0 = Live, Class 1 = Spoof (or vice versa depending on your generator)
    is_live = score < THRESHOLD
    label = "LIVE ✅" if is_live else "SPOOF ❌"
    confidence = (1 - score) if is_live else score
    
    print("\nPrediction")
    print("-" * 22)
    print(f"Label      : {label}")
    print(f"Score      : {score:.4f}")
    print(f"Confidence : {confidence * 100:.2f}%")

if __name__ == "__main__":
    IMAGE_PATH = Path("test_images/sample_fingerprint.jpg")
    if IMAGE_PATH.exists():
        predict_image(IMAGE_PATH)
    else:
        print(f"Please place a test image at {IMAGE_PATH} to run inference.")
        
# """
# Fingerprint Liveness Detection
# Inference Script

# Predict whether a fingerprint image is:

# - Live
# - Spoof
# """

# from pathlib import Path

# import cv2
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt

# # ==========================================================
# # Configuration
# # ==========================================================

# MODEL_PATH = "models/liveness_model.keras"

# THRESHOLD_PATH = "outputs/best_threshold.txt"

# IMG_SIZE = (224, 224)

# IMAGE_PATH = "test_images/1__M_Left_index_finger.BMP"     # Change this image


# # ==========================================================
# # Load Model
# # ==========================================================

# print("Loading model...")

# model = tf.keras.models.load_model(MODEL_PATH)

# print("Model loaded.")


# # ==========================================================
# # Load Threshold
# # ==========================================================

# with open(THRESHOLD_PATH, "r") as f:
#     threshold = float(f.read())

# print(f"Threshold: {threshold:.4f}")


# # ==========================================================
# # Load Image
# # ==========================================================

# image = cv2.imread(IMAGE_PATH)

# if image is None:
#     raise FileNotFoundError(
#         f"Cannot read image: {IMAGE_PATH}"
#     )

# image_rgb = cv2.cvtColor(
#     image,
#     cv2.COLOR_BGR2RGB
# )


# # ==========================================================
# # Preprocess (same as training)
# # ==========================================================

# input_image = cv2.resize(
#     image_rgb,
#     IMG_SIZE
# )

# input_image = input_image.astype(np.float32)

# # Normalize to [0,1]
# input_image = input_image / 255.0

# # Same ImageNet normalization used during training
# IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
# IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# input_image = (input_image - IMAGENET_MEAN) / IMAGENET_STD

# input_image = np.expand_dims(
#     input_image,
#     axis=0
# )

# # ==========================================================
# # Predict
# # ==========================================================

# score = model.predict(
#     input_image,
#     verbose=0
# )[0][0]

# prediction = int(score >= threshold)

# confidence = score if prediction == 1 else 1 - score

# label = "SPOOF ❌" if prediction == 1 else "LIVE ✅"


# print("\nPrediction")
# print("------------------------")
# print(f"Label      : {label}")
# print(f"Score      : {score:.4f}")
# print(f"Confidence : {confidence:.2%}")


# # ==========================================================
# # Display Image
# # ==========================================================

# plt.figure(figsize=(5, 5))

# plt.imshow(image_rgb)

# plt.axis("off")

# plt.title(
#     f"{label}\n"
#     f"Spoof Probability: {score:.3f}\n"
#     f"Confidence: {confidence:.2%}"
# )
# plt.show()