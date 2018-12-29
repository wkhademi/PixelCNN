import matplotlib
matplotlib.use('Agg')

import os
import sys
import tensorflow as tf
import numpy as np
from scipy.io import loadmat
from urllib2 import urlopen
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data
from network import Network


def unpickle(file):
    """
        Helper function for loading in CIFAR-10 Image Dataset.

        Args:
            file: filename in the form of a string

        Returns:
            dict: A python dictionary containing {data, labels, batch_label, filenames}
    """
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def split_data(images):
    train_images, test_images = train_test_split(images, test_size=0.2, shuffle=True)

    return train_images, test_images


def retrieve_data(config):
    """
        Load Dataset of Images.

        If '--CIFAR' flag is used, load the CIFAR10 Dataset.
        If '--MNIST' flag is used, load the MNIST Dataset.
        If '--FREY' flag is used, load the FREY Dataset

        CIFAR10 Dataset downloaded from:
        https://www.cs.toronto.edu/~kriz/cifar.html

        Returns:
            images: A set of images to train and test a PixelCNN on.
    """
    if (config == '--CIFAR'):
        image_set = unpickle('cifar-10-batches-py/data_batch_2')

        # reshape to correct image structure
        images = image_set['data'].reshape((len(image_set['data']), 3, 32, 32)).transpose(0, 2, 3, 1)

        # split dataset into train and test sets
        train_images, test_images = split_data(images)

    elif (config == '--MNIST'):
        image_set = input_data.read_data_sets('MNIST_data/', one_hot=False)

        # reshape to correct image structure
        train_images = np.reshape(image_set.train.images, (-1, 28, 28, 1))
        test_images = np.reshape(image_set.test.images, (-1, 28, 28, 1))

    elif (config == '--FREY'):
        url = 'https://cs.nyu.edu/~roweis/data/frey_rawface.mat'
        data_filename = os.path.basename(url)

        if not os.path.exists(data_filename):
            file = urlopen(url)

            with open(os.path.basename(url), 'wb') as local_file:
                local_file.write(file.read())

        images = loadmat(data_filename, squeeze_me=True, struct_as_record=False)
        images = images["ff"].T.reshape((-1, 28, 20, 1))

        # split dataset into train and test sets
        train_images, test_images = split_data(images)

    return train_images, test_images


def run(config):
    train_images, test_images = retrieve_data(config)

    if (config == '--CIFAR'):
        height, width, channels = (32, 32, 3)
    elif (config == '--MNIST'):
        height, width, channels = (28, 28, 1)
    elif (config == '--FREY'):
        height, width, channels = (28, 20, 1)

    network = Network(train_images, test_images, height, width, channels, config)

    network.train()
    network.test()


if __name__ == '__main__':
    config = sys.argv[1]
    run(config)
