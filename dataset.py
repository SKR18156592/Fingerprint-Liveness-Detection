import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 8

# Load Training Dataset with Augmentation
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset_split/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True
)

# Load Validation Dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset_split/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

# Load Test Dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset_split/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

# Optional Data Augmentation Layer for Training
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomBrightness(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

def preprocess(image, label):
    # Rescale or normalize if needed (MobileNetV3 handles its own scaling)
    return image, label

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# import tensorflow as tf

# IMG_SIZE = (224,224)
# BATCH_SIZE = 8

# train_ds = tf.keras.utils.image_dataset_from_directory(
#     "dataset_split/train",
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     label_mode="binary",
#     shuffle=True,
#     seed=42
# )

# val_ds = tf.keras.utils.image_dataset_from_directory(
#     "dataset_split/val",
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     label_mode="binary",
#     shuffle=False
# )

# test_ds = tf.keras.utils.image_dataset_from_directory(
#     "dataset_split/test",
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     label_mode="binary",
#     shuffle=False
# )

# # -------------------------
# # Data Augmentation
# # -------------------------

# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.RandomRotation(0.083),
#     # tf.keras.layers.RandomFlip("horizontal"),
#     tf.keras.layers.RandomBrightness(0.15),
#     tf.keras.layers.RandomContrast(0.15),
# ])

# def augment(image, label):
#     image = data_augmentation(image)
#     return image, label

# train_ds = train_ds.map(augment)

# # -------------------------
# # Normalization
# # -------------------------

# IMAGENET_MEAN = tf.constant([0.485,0.456,0.406], dtype=tf.float32)
# IMAGENET_STD = tf.constant([0.229,0.224,0.225], dtype=tf.float32)

# def normalize(image, label):
#     image = tf.cast(image, tf.float32) / 255.0
#     image = (image - IMAGENET_MEAN) / IMAGENET_STD
#     return image, label

# train_ds = train_ds.map(normalize)
# val_ds = val_ds.map(normalize)
# test_ds = test_ds.map(normalize)

# AUTOTUNE = tf.data.AUTOTUNE

# train_ds = train_ds.prefetch(AUTOTUNE)
# val_ds = val_ds.prefetch(AUTOTUNE)
# test_ds = test_ds.prefetch(AUTOTUNE)