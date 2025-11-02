import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================================================
#  STEP 1: Define dataset paths
# =========================================================
train_dir = r"C:\Users\HP\Desktop\major project\train"
val_dir = r"C:\Users\HP\Desktop\major project\val"

if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    raise FileNotFoundError("❌ Verify that 'train' and 'val' folders exist in your major project directory.")

# =========================================================
#  STEP 2: Preprocess Images
# =========================================================
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
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# =========================================================
#  STEP 3: Build CNN model
# =========================================================
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

    Dense(3, activation='softmax')  # 3 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# =========================================================
#  STEP 4: Train model with EarlyStopping
# =========================================================
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("🚀 Training model... please wait.")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[early_stop],
    verbose=1
)

# =========================================================
#  STEP 5: Save model in modern format
# =========================================================
model.save("potato_disease_model.keras")
print("✅ Model saved successfully as potato_disease_model.keras")

# =========================================================
#  STEP 6: Evaluate model
# =========================================================
val_loss, val_acc = model.evaluate(val_data)
print(f"📊 Validation Accuracy: {val_acc*100:.2f}% | Loss: {val_loss:.4f}")

# =========================================================
#  STEP 7: Plot Accuracy Graph
# =========================================================
plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# =========================================================
#  STEP 8: Predict on All Images in a Folder
# =========================================================
predict_dir = r"C:\Users\HP\Desktop\major project\predict"  # Folder with test images
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
SUGGESTIONS = {
    "Early Blight": "🛡 Use fungicides like chlorothalonil or mancozeb. Remove infected leaves and rotate crops.",
    "Late Blight": "🧪 Apply copper-based fungicides. Avoid overhead watering and improve air circulation.",
    "Healthy": "✅ No action needed. Maintain regular monitoring and good soil health."
}

image_paths = glob.glob(os.path.join(predict_dir, "*.jpg")) + glob.glob(os.path.join(predict_dir, "*.png"))

if not image_paths:
    print("⚠️ No images found in the predict folder.")
else:
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0])) * 100

        print(f"\n🖼 Image: {os.path.basename(img_path)}")
        print(f"🔍 Predicted Disease: {predicted_class}")
        print(f"📈 Confidence: {confidence:.2f}%")
        print(f"🧾 Recommendation: {SUGGESTIONS[predicted_class]}")

        # Show image with prediction
        plt.figure(figsize=(4,4))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{predicted_class} ({confidence:.2f}%)")
        plt.show()