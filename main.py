import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Lambda, Concatenate
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

INPUT_SHAPE = (160, 160, 3)
EMBEDDING_DIM = 128
BATCH_SIZE = 32
EPOCHS = 50
MARGIN = 0.3
LFW_PATH = "./lfw"

OPT = 'use'

@tf.keras.utils.register_keras_serializable()
def l2_normalize_layer(y):
    return tf.math.l2_normalize(y, axis=1)

def create_embedding_model(input_shape=INPUT_SHAPE, embedding_dim=EMBEDDING_DIM):
    base_model = MobileNetV2(include_top=False, input_shape=input_shape, weights='imagenet')
    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024)(x)
    x = layers.Activation('tanh')(x)
    x = layers.Dropout(0.5)(x)
    x = Dense(1024)(x)
    x = layers.Activation('tanh')(x)
    x = layers.Dropout(0.5)(x)
    x = Dense(embedding_dim)(x)
    x = Lambda(l2_normalize_layer, output_shape=(embedding_dim,))(x)
    model = Model(inputs, x)
    return model

def triplet_accuracy(margin=MARGIN):
    def metric(y_true, y_pred):
        anchor, positive, negative = y_pred[:, 0:EMBEDDING_DIM], y_pred[:, EMBEDDING_DIM:2 * EMBEDDING_DIM], y_pred[:, 2 * EMBEDDING_DIM:]
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        return tf.reduce_mean(tf.cast(pos_dist + margin < neg_dist, tf.float32))
    return metric


def triplet_loss(margin=MARGIN):
    def loss(y_true, y_pred):
        anchor, positive, negative = y_pred[:, 0:EMBEDDING_DIM], y_pred[:, EMBEDDING_DIM:2 * EMBEDDING_DIM], y_pred[:,
                                                                                                             2 * EMBEDDING_DIM:]
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        basic_loss = pos_dist - neg_dist + margin
        return tf.reduce_mean(tf.maximum(basic_loss, 0.0))

    return loss


def create_triplet_model(embedding_model, input_shape=INPUT_SHAPE):
    anchor_input = Input(shape=input_shape, name='anchor_input')
    positive_input = Input(shape=input_shape, name='positive_input')
    negative_input = Input(shape=input_shape, name='negative_input')

    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)

    merged_output = Concatenate(axis=1)([anchor_embedding, positive_embedding, negative_embedding])
    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_output)
    return model


def load_lfw_triplets(lfw_path, input_shape=INPUT_SHAPE):
    person_dirs = [os.path.join(lfw_path, d if isinstance(d, str) else d.decode('utf-8'))
                   for d in os.listdir(lfw_path)
                   if os.path.isdir(os.path.join(lfw_path, d if isinstance(d, str) else d.decode('utf-8')))]
    triplets = []

    for person_dir in person_dirs:
        images = [os.path.join(person_dir, f if isinstance(f, str) else f.decode('utf-8'))
                  for f in os.listdir(person_dir)
                  if (f if isinstance(f, str) else f.decode('utf-8')).lower().endswith('.jpg')]
        if len(images) < 2:
            continue

        for i in range(len(images) - 1):
            anchor = images[i]
            positive = images[i + 1]

            negative_person_dir = np.random.choice([d for d in person_dirs if d != person_dir])
            negative_images = [os.path.join(negative_person_dir, f if isinstance(f, str) else f.decode('utf-8'))
                               for f in os.listdir(negative_person_dir)
                               if (f if isinstance(f, str) else f.decode('utf-8')).lower().endswith('.jpg')]
            if not negative_images:
                continue
            negative = np.random.choice(negative_images)

            triplets.append((anchor, positive, negative))
    return triplets


def preprocess_image(filepath):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, INPUT_SHAPE[:2])
    img = img / 255.0
    return img


def triplet_generator(triplets, batch_size=BATCH_SIZE):
    while True:
        np.random.shuffle(triplets)
        for i in range(0, len(triplets), batch_size):
            batch_triplets = triplets[i:i + batch_size]
            anchor_batch = []
            positive_batch = []
            negative_batch = []
            for anchor_path, positive_path, negative_path in batch_triplets:
                anchor_batch.append(preprocess_image(anchor_path))
                positive_batch.append(preprocess_image(positive_path))
                negative_batch.append(preprocess_image(negative_path))
            yield (tf.stack(anchor_batch), tf.stack(positive_batch), tf.stack(negative_batch)), np.zeros(len(anchor_batch))

def get_embedding(model_path, image_path):
    embedding_model = tf.keras.models.load_model(model_path, compile=False)

    img = preprocess_image(image_path)
    img = tf.expand_dims(img, axis=0)

    embedding = embedding_model.predict(img)[0]

    return embedding


if OPT == 'train':
    embedding_model = create_embedding_model()
    triplet_model = create_triplet_model(embedding_model)
    triplet_model.compile(optimizer=Adam(0.0001), loss=triplet_loss(), metrics=[triplet_accuracy()])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    triplets = load_lfw_triplets(LFW_PATH)
    train_triplets, val_triplets = train_test_split(triplets, test_size=0.1, random_state=42)

    train_gen = triplet_generator(train_triplets, batch_size=BATCH_SIZE)
    val_gen = triplet_generator(val_triplets, batch_size=BATCH_SIZE)

    steps_per_epoch = len(train_triplets) // BATCH_SIZE
    validation_steps = len(val_triplets) // BATCH_SIZE

    triplet_model.fit(train_gen, validation_data=val_gen, steps_per_epoch=steps_per_epoch,
                  validation_steps=validation_steps, epochs=EPOCHS, callbacks=[early_stopping])

    embedding_model.save("face_embedding_model.keras")
    print("Embedding model saved!")

elif OPT == 'use':
    image_path_1 = './test/art.png'
    image_path_2 = './test/art2.png'
    embedding1 = get_embedding('face_embedding_model.keras', image_path_1)
    embedding2 = get_embedding('face_embedding_model.keras', image_path_2)
    print("Embedding vector 1 (128-d):", embedding1)
    print("Embedding vector 2 (128-d):", embedding2)
    print("Embedding vector diff (128-d):", embedding1 - embedding2)
