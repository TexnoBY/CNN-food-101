"""This module implements data feeding and training loop to create model
to classify X-Ray chest images as a lab example for BSU students.
"""

__author__ = 'Alexander Soroka, soroka.a.m@gmail.com'
__copyright__ = """Copyright 2020 Alexander Soroka"""


import argparse
import glob
import numpy as np
import tensorflow as tf
import time
from tensorflow.python import keras as keras
from tensorflow.python.keras.callbacks import LearningRateScheduler

# Avoid greedy memory allocation to allow shared GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


LOG_DIR = 'logs4'
BATCH_SIZE = 16
NUM_CLASSES = 101
RESIZE_TO = 224
TRAIN_SIZE = 101000


def parse_proto_example(proto):
  keys_to_features = {
    'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
    'image/label': tf.io.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
  }
  example = tf.io.parse_single_example(proto, keys_to_features)
  example['image'] = tf.image.decode_jpeg(example['image/encoded'], channels=3)
  example['image'] = tf.image.convert_image_dtype(example['image'], dtype=tf.uint8)
  example['image'] = tf.image.resize(example['image'], tf.constant([250, 250]), method='nearest')
  return example['image'], tf.one_hot(example['image/label'], depth=NUM_CLASSES)


def normalize(image, label):
  return tf.image.per_image_standardization(image), label

def random_crop(image,label):
      return tf.image.random_crop(image,[RESIZE_TO, RESIZE_TO, 3]),label

def create_dataset(filenames, batch_size):
  """Create dataset from tfrecords file
  :tfrecords_files: Mask to collect tfrecords file of dataset
  :returns: tf.data.Dataset
  """
  return tf.data.TFRecordDataset(filenames)\
    .map(parse_proto_example, num_parallel_calls=tf.data.AUTOTUNE)\
    .map(normalize)\
    .batch(batch_size)\
    .prefetch(tf.data.AUTOTUNE)


def build_model(mode):
  inputs = tf.keras.Input(shape=(250, 250, 3))
  aug_data = tf.keras.layers.experimental.preprocessing.RandomCrop(224, 224)(inputs)
  aug_data = tf.keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal")(aug_data)
  aug_data = tf.keras.layers.experimental.preprocessing.RandomRotation(0.1, fill_mode='reflect')(aug_data)
  x = tf.keras.applications.EfficientNetB0(include_top=False,
                                           weights='imagenet',
                                           input_tensor=aug_data)
  x.trainable = False
  x = tf.keras.layers.GlobalAveragePooling2D()(x.output)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)


def main():
  args = argparse.ArgumentParser()
  args.add_argument('--train', type=str, help='Glob pattern to collect train tfrecord files, use single quote to escape *')
  args = args.parse_args()

  dataset = create_dataset(glob.glob(args.train), BATCH_SIZE)
  #for x, y in dataset.take(1):
   #     print(x)
  train_size = int(TRAIN_SIZE * 0.7 / BATCH_SIZE)
  train_dataset = dataset.take(train_size)
  validation_dataset = dataset.skip(train_size)


  for mode in ['all']:

      model = build_model(mode)

      model.compile(
        optimizer=tf.optimizers.Adam(lr=0.0001),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=[tf.keras.metrics.categorical_accuracy],
      )

      log_dir='{}/augmentation_all_{}-{}'.format(LOG_DIR, mode, time.time())
      model.fit(
        train_dataset,
        epochs=15,
        validation_data=validation_dataset,
        callbacks=[
          tf.keras.callbacks.TensorBoard(log_dir),
        ]
      )


if __name__ == '__main__':
    main()