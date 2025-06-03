import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import cv2

class ImageLoader(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size=32, img_size=(64, 64), augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = False

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.image_paths))
        if start >= len(self.image_paths):
            raise IndexError("Batch index out of range")
        batch_paths = self.image_paths[start:end]
        batch_labels = self.labels[start:end]
        batch_images = []
        for path in batch_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            img = img.astype('float32') / 255.0
            batch_images.append(img)
        return np.array(batch_images), np.array(batch_labels)