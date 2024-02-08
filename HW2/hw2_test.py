import tensorflow as tf
import os
import pickle
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import Input, layers

import sys

import hw2_complete as hw

model1 = hw.build_model1()
model2 = hw.build_model2()
model3 = hw.build_model3()
#model50k = hw.build_model50k()

# # Load CIFAR 10 dataset
# (train_images, train_labels), (test_images, test_labels) = \
#     tf.keras.datasets.cifar10.load_data()
#
# train_labels = train_labels.squeeze()
# test_labels = test_labels.squeeze()
#
# train_images = train_images / 255.0
# test_images = test_images / 255.0

def load_cifar10_data(file):
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict[b'data'], cifar_dict[b'labels']

    # Assuming you've extracted the CIFAR-10 dataset to 'cifar-10-batches-py' directory


cifar10_dir = r'C:\Users\X390 Yoga\Desktop\Swayam\Intro to ML\cifar-10-python\cifar-10-batches-py'
# cifar10_dir = r'C:\Users\SirM\Desktop\Swayam\Intro to ML\cifar-10-batches-py'
training_files = [os.path.join(cifar10_dir, 'data_batch_{}'.format(i)) for i in range(1, 6)]
test_file = os.path.join(cifar10_dir, 'test_batch')

train_images, train_labels = [], []
for file in training_files:
    data, labels = load_cifar10_data(file)
    train_images.append(data)
    train_labels += labels

train_images = np.concatenate(train_images, axis=0).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
train_labels = np.array(train_labels)

test_data, test_labels = load_cifar10_data(test_file)
test_images = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
test_labels = np.array(test_labels)

# Normalize pixel values
train_images, test_images = train_images / 255.0, test_images / 255.0

try:
    model50k = tf.keras.models.load_model("best_model.h5")
except:
    print("Failure loading best_model.h5")
print("Model50k loaded")
model3.summary()


def count_layers(model, layer_type):
    lyr_count = 0
    for l in model.layers:
        if l.__class__.__name__ == layer_type:
            lyr_count += 1
    return lyr_count


def test_model1_params():
    assert model1.count_params() == 704842


def test_model1_convs():
    n_convs = count_layers(model1, "Conv2D")
    assert n_convs == 7


def test_model1_batchnorm():
    n_bn = count_layers(model1, "BatchNormalization")
    assert n_bn in [7, 8]


def test_model1_dense():
    n_dense = count_layers(model1, "Dense")
    assert n_dense == 2


def test_model2_params():
    # should really be 104106 if the model is built correctly.  Leaving in 104138
    # for compatibility with earlier (incorrect) test script
    assert model2.count_params() in [104106, 104138]


def test_model3_params():
    assert model3.count_params() == 709066


def test_model3_adds():
    n_adds = count_layers(model3, "Add")
    assert n_adds == 3


def test_model50k_params():
    assert model50k.count_params() <= 50000


def test_model50k_acc50():
    loss, acc = model50k.evaluate(test_images, test_labels)
    assert acc >= 0.5


def test_model50k_acc60():
    loss, acc = model50k.evaluate(test_images, test_labels)
    assert acc >= 0.60