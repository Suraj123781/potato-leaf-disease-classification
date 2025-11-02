import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Disable GPU for Render
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Dataset paths (relative)
train_dir = os.path.join("data", "train")
val_dir = os.path.join("data", "val")

if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    raise FileNotFoundError("❌ 'data/train' or 'data/val' folder is missing.")

# Preprocessing
img_size = (128, 128)
batch_size = 32
epochs = 25

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)
val_data = val_datagen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)

# Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
print("🚀 Training model...")
history = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=[early_stop], verbose=1)

# Save model
model.save("potato_disease_model.keras")
print("✅ Model saved as potato_disease_model.keras")

# Evaluate
val_loss, val_acc = model.evaluate(val_data)
print(f"📊 Validation Accuracy: {val_acc*100:.2f}% | Loss: {val_loss:.4f}")

# Plot
plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()