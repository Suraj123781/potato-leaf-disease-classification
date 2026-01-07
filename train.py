import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

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

# Get predictions for the validation set
print("\nGenerating predictions and calculating metrics...")

def get_labels_and_predictions(dataset, model, class_names):
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    for images, labels in dataset:
        # Get true labels
        y_true.extend(tf.argmax(labels, axis=1).numpy())
        
        # Get predictions and probabilities
        probs = model.predict(images, verbose=0)
        y_pred.extend(tf.argmax(probs, axis=1).numpy())
        y_pred_proba.extend(np.max(probs, axis=1))
    
    return np.array(y_true), np.array(y_pred), np.array(y_pred_proba)

# Get true labels and predictions
y_true, y_pred, y_pred_proba = get_labels_and_predictions(val_ds, model, class_names)

# Calculate and print classification report
print("\n" + "="*50)
print("Classification Report")
print("="*50)
print(classification_report(y_true, y_pred, target_names=class_names))

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix saved as 'confusion_matrix.png'")

# Calculate and display confidence scores
confidence_scores = []
for i in range(len(y_true)):
    confidence_scores.append({
        'True_Label': class_names[y_true[i]],
        'Predicted_Label': class_names[y_pred[i]],
        'Confidence_Score': f"{y_pred_proba[i]:.4f}",
        'Correct': y_true[i] == y_pred[i]
    })

# Convert to DataFrame for better display
confidence_df = pd.DataFrame(confidence_scores)
print("\nSample of confidence scores:")
print(confidence_df.head())

# Save confidence scores to CSV
confidence_df.to_csv('confidence_scores.csv', index=False)
print("\nFull confidence scores saved to 'confidence_scores.csv'")

# Calculate and display average confidence for correct and incorrect predictions
correct_conf = y_pred_proba[y_true == y_pred]
incorrect_conf = y_pred_proba[y_true != y_pred]

print(f"\nAverage confidence for correct predictions: {np.mean(correct_conf):.4f}")
print(f"Average confidence for incorrect predictions: {np.mean(incorrect_conf):.4f}" if len(incorrect_conf) > 0 else "No incorrect predictions to calculate average confidence")

print("\n✅ Evaluation complete! Check the generated files for detailed metrics.")
