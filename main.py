import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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


def retrieve_data():
	"""
		Load CIFAR-10 Image Dataset.

		Dataset downloaded from:
		https://www.cs.toronto.edu/~kriz/cifar.html

		Returns:
			images: A set of images to train and test a PixelCNN on.
	"""
	image_set = unpickle('cifar-10-batches-py/data_batch_2')

	# reshape to correct image structure
	images = image_set['data'].reshape((len(image_set['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
	#plt.imshow(images[0], interpolation='nearest')
	#plt.show()

	# normalize images to have every pixel be between [-1, 1]
	images = 2 * (images / 255.0) - 1

	return images


def run():
	images = retrieve_data()


if __name__ == '__main__':
	run()
