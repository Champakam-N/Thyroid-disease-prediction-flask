# =========================================
# Thyroid CNN Training Script (3 Classes)
# =========================================

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json

# ======================
# Dataset Path
# ======================
dataset_dir = r"D:\champa\projects\Thyroid Data\dataset"

# ======================
# Data Augmentation
# ======================
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
    validation_split=0.2  # 80% train, 20% validation
)

train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(128,128),
    batch_size=16,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(128,128),
    batch_size=16,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# ======================
# Save class indices for correct mapping
# ======================
with open("class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)
print("Class indices saved:", train_generator.class_indices)

# ======================
# Build CNN Model
# ======================
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
    Dense(3, activation='softmax')  # 3-class output
])

# ======================
# Compile Model
# ======================
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ======================
# Callbacks to Save Best Model
# ======================
checkpoint = ModelCheckpoint(
    "thyroid_cnn_model_3class.h5", monitor='val_accuracy', save_best_only=True, verbose=1
)
early_stop = EarlyStopping(
    monitor='val_loss', patience=5, verbose=1, restore_best_weights=True
)

# ======================
# Train Model
# ======================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=25,
    callbacks=[checkpoint, early_stop]
)

print("Training completed. Model saved as 'thyroid_cnn_model_3class.h5'")
