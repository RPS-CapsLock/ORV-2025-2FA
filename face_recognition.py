import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.utils import compute_class_weight
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import cv2
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
import joblib
from tqdm import tqdm
from util.image_loader import ImageLoader

IMG_SIZE = (128, 128)
BATCH_SIZE = 5

def load_data(mismatch_path, data_path):
    image_paths, labels = [], []
    for img_file in os.listdir(mismatch_path):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm')):
            image_paths.append(os.path.join(mismatch_path, img_file))
            labels.append(0)
    for img_file in os.listdir(data_path):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm')):
            image_paths.append(os.path.join(data_path, img_file))
            labels.append(1)
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels)
    return X_train, X_val, y_train, y_val

def build_model():
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
    inputs = layers.Input(shape=input_shape)
    base_model = MobileNetV2(include_top=False, input_shape=input_shape, weights='imagenet')
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs=inputs, outputs=x)

def train_and_evaluate(user_id, data_path, mismatch_path = './data/preccessed/kombinirano'):
    print(f"\nTraining model for user: {user_id}")
    X_train, X_val, y_train, y_val = load_data(mismatch_path, data_path)

    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

    train_loader = ImageLoader(X_train, y_train, BATCH_SIZE, IMG_SIZE, augment=True)
    val_loader = ImageLoader(X_val, y_val, BATCH_SIZE, IMG_SIZE, augment=False)

    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                  loss='binary_crossentropy', metrics=['accuracy'])

    callbacks = [TensorBoard(log_dir=f'logs/{user_id}'),
                 EarlyStopping(patience=10, restore_best_weights=True),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)]

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    steps_per_epoch = len(train_loader)
    validation_steps = len(val_loader)
    print(f"Train batches: {steps_per_epoch}, Validation batches: {validation_steps}")

    model.fit(train_loader, validation_data=val_loader, epochs=150,
              callbacks=callbacks, class_weight=class_weight_dict,
              steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

    model.save(f'm_{user_id}.keras')
    print(f"Model saved as m_{user_id}.keras")

    val_preds, val_labels = [], []
    for batch_x, batch_y in tqdm(val_loader, total=validation_steps, desc="Predicting validation batches"):
        preds = model.predict(batch_x, verbose=0).flatten()
        val_preds.extend(preds)
        val_labels.extend(batch_y)

    if len(val_preds) == 0:
        print("No valid predictions made. Check data or model.")
        return

    val_preds = np.array(val_preds)
    val_labels = np.array(val_labels)

    fpr, tpr, thresholds = roc_curve(val_labels, val_preds)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    if optimal_threshold < 0.5:
        optimal_threshold = 0.5
    print(f"Optimal threshold: {optimal_threshold}")
    joblib.dump(optimal_threshold, f'm_{user_id}_threshold.pkl')
    print(f"Saved threshold as m_{user_id}_threshold.pkl")

def predict(image_path, user_id):
    model = models.load_model(f'm_{user_id}.keras')
    try:
        optimal_threshold = joblib.load(f'm_{user_id}_threshold.pkl')
        print(f"Using saved threshold: {optimal_threshold}")
    except:
        optimal_threshold = 0.5
        print("Threshold file not found. Using default threshold 0.5")

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]
    label = 'match' if pred > optimal_threshold else 'mismatch'
    confidence = pred if pred > optimal_threshold else 1 - pred
    print(f"Prediction: {label} with confidence: {confidence:.2f}")

def train(user_id):
    data_path = f'./{user_id}'
    train_and_evaluate(user_id, data_path)

def use(user_id, test_p):
    predict(user_id, test_p)