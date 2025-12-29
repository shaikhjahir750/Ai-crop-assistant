import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# ==============================
# GPU Check
# ==============================
if tf.config.list_physical_devices('GPU'):
    print(f"✅ GPU detected: {tf.config.list_physical_devices('GPU')[0]}")
else:
    print("⚠️ GPU not detected, using CPU")

# Limit TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==============================
# Paths
# ==============================
train_dir = "dataset/train"
val_dir = "dataset/validation"
MODEL_PATH = "models/disease_model_gpu.h5"

# ==============================
# Data Augmentation
# ==============================
img_size = (128, 128)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

num_classes = len(train_generator.class_indices)
print(f"🌿 Classes detected: {num_classes} -> {list(train_generator.class_indices.keys())}")

# ==============================
# CNN Model
# ==============================
model = Sequential([
    tf.keras.Input(shape=(*img_size, 3)),
    Conv2D(32, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ==============================
# Training
# ==============================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, save_best_only=True)
]

with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=25,
        callbacks=callbacks
    )

# ==============================
# Save model
# ==============================
model.save(MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")

# ==============================
# Plot training
# ==============================
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend(); plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title('Loss')

plt.show()
