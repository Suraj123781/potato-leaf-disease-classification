import tensorflow as tf
from tensorflow.keras import layers, models
import os

# ✅ DATASET PATHS (YOUR PATHS)
TRAIN_DIR = r"D:\potato project\data_resized\train"
VAL_DIR = r"D:\potato project\data_resized\val"

# ✅ IMAGE SETTINGS
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ✅ LOAD DATA
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

# ✅ CLASS NAMES
class_names = train_ds.class_names
print("Detected Classes:", class_names)

# ✅ PERFORMANCE OPTIMIZATION
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ✅ DATA AUGMENTATION
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ✅ LOAD PRETRAINED MODEL (TRANSFER LEARNING)
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    input_shape=IMG_SIZE + (3,),
    weights="imagenet"
)

base_model.trainable = False  # freeze base model

# ✅ BUILD FINAL MODEL
inputs = layers.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = models.Model(inputs, outputs)

# ✅ COMPILE MODEL
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ✅ TRAIN MODEL
EPOCHS = 10

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ✅ SAVE MODEL
os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/potato_model.h5"
model.save(MODEL_PATH)

print("✅ Model training complete!")
print("✅ Model saved at:", MODEL_PATH)

# ✅ EVALUATE MODEL
val_loss, val_acc = model.evaluate(val_ds)
print(f"✅ Validation Accuracy: {val_acc * 100:.2f}%")
