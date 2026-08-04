import tensorflow as tf

base_model = tf.keras.applications.MobileNetV3Small(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)

# base_model.trainable = False

# Unfreeze the last 20–30 layers
base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False

inputs = tf.keras.Input(shape=(224,224,3))

x = base_model(inputs, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)


# ==========================================================
# Callbacks
# ==========================================================

from pathlib import Path

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_DIR / "liveness_model.keras",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

csv_logger = tf.keras.callbacks.CSVLogger(
    OUTPUT_DIR / "training_log.csv"
)

# ==========================================================
# Train Model
# ==========================================================
from  dataset import train_ds,val_ds

EPOCHS = 20

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr,
        csv_logger
    ]
)

# ==========================================================
# Save Final Model
# ==========================================================

model.save(MODEL_DIR / "liveness_model.keras")

print("\nTraining Complete!")
print("Model saved to:", MODEL_DIR / "liveness_model.keras")