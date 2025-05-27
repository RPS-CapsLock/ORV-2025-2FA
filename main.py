import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import albumentations as A
import cv2
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2

if len(sys.argv) < 3 or len(sys.argv) > 4:
    print("Usage: python script.py <OPT> <USERID> <TEST>")
    sys.exit(1)

OPT = sys.argv[1]
USERID = sys.argv[2]
TEST = ""
if OPT == 'use':
    TEST = sys.argv[3]

DATA_PATH = f'./{USERID}'
MISMATCH_PATH = f'./mismatch'
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_CLASSES = 2

class ImageLoader(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size=32, img_size=(64, 64), augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment

        self.augmentation = A.Compose([
            A.Rotate(limit=10),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, p=0.2),
        ])

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_images = []
        for path in batch_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)

            if self.augment:
                img = self.augmentation(image=img)['image']

            img = img.astype('float32') / 255.0
            batch_images.append(img)

        return np.array(batch_images), np.array(batch_labels)

def load_data():
    image_paths = []
    labels = []

    class_path = MISMATCH_PATH
    for img_file in os.listdir(class_path):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm')):
            image_paths.append(os.path.join(class_path, img_file))
            labels.append(0)

    class_path = DATA_PATH
    for img_file in os.listdir(class_path):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm')):
            image_paths.append(os.path.join(class_path, img_file))
            labels.append(1)

    print("------------------------------------------------------------")
    print(image_paths)
    print(labels)
    print("------------------------------------------------------------")

    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    return X_train, X_val, y_train, y_val

def build_model():
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
    inputs = layers.Input(shape=input_shape)
    x = inputs

    base_model = MobileNetV2(include_top=False, input_shape=input_shape, weights='imagenet')
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    x = base_model(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inputs, outputs=x)
    return model

def train_and_evaluate():
    print(f"\nTraining model for user: {USERID}")
    X_train, X_val, y_train, y_val = load_data()

    train_loader = ImageLoader(X_train, y_train, BATCH_SIZE, IMG_SIZE, augment=True)
    val_loader = ImageLoader(X_val, y_val, BATCH_SIZE, IMG_SIZE, augment=False)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        TensorBoard(log_dir=f'logs/{USERID}'),
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]

    history = model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=50,
        callbacks=callbacks
    )

    model.save(f'm_{USERID}.keras')
    print(f"Model saved as m_{USERID}.keras")

def predict(image_path):
    model = models.load_model(f'm_{USERID}.keras')
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]
    label = 'match' if pred > 0.5 else 'mismatch'
    confidence = pred if pred > 0.5 else 1 - pred
    print(f"Prediction: {label} with confidence: {confidence:.2f}")

if OPT == 'train':
    train_and_evaluate()
elif OPT == 'use':
    predict(TEST)
else:
    print("Invalid OPT. Use 'train' or 'use'.")
