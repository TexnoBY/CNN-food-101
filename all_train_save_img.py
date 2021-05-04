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
import tensorflow_addons as tfa
from tensorflow.python.keras.callbacks import LearningRateScheduler

from PIL import Image


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
  example['image'] = tf.image.random_crop(example['image'],[RESIZE_TO, RESIZE_TO, 3])
  example['image'] = tf.image.flip_left_right(example['image'])
  example['image'] = tfa.image.rotate(example['image'], 0.4, fill_mode = 'reflect', fill_value = 0)
  return example['image'], tf.one_hot(example['image/label'], depth=NUM_CLASSES)

def normalize(image, label):
  return tf.image.per_image_standardization(image), label


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

def main():
  args = argparse.ArgumentParser()
  args.add_argument('--train', type=str, help='Glob pattern to collect train tfrecord files, use single quote to escape *')
  args = args.parse_args()

  dataset = create_dataset(glob.glob(args.train), BATCH_SIZE)
  q = 0
  for x, y in dataset.take(40):
    for j in x:

      q += 1
      img = Image.fromarray(j.numpy(), 'RGB')
      img.save(f'all-{q}.jpg')
      break

if __name__ == '__main__':
    main()
