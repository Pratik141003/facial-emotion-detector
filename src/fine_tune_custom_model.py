import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# === SETTINGS ===
dataset_path = "facial-emotion-detector\dataset"
img_size = (48, 48)
batch_size = 32
epochs = 20

# === DATA LOADING ===
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# === CLASS WEIGHTS ===
labels = train_gen.classes
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(class_weights))

# === MODEL DEFINITION (MobileNet-based) ===
base_model = MobileNetV2(input_shape=(48, 48, 3), include_top=False, weights='imagenet')
model = Sequential([
    tf.keras.layers.Resizing(48, 48),  # Ensure correct input
    tf.keras.layers.Conv2D(3, (3, 3), padding='same'),  # Convert grayscale to 3 channels
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# === TRAIN ===
checkpoint = ModelCheckpoint("models/emotion_model_custom.h5", monitor="val_accuracy", save_best_only=True, verbose=1)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=[checkpoint]
)
