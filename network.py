import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Network:
	def __init__(self,
			train_inputs,
			test_inputs,
			height,
			width,
			channels):
		self.train_inputs = train_inputs
		self.test_inputs = test_inputs
		self.height = height
		self.width = width
		self.channels = channels
		self.num_residuals = 3
		self.network = None


	def build_network(self,
					inputs,
					is_training,
					dropout):
		pixelCNNModel = PixelCNN(inputs, self.height, self.width, self.channels)

		# First layer in the network
		kernel_shape = [7, 7, self.channels, 32]
		bias_shape = [32]
		strides = [1, 1, 1, 1]
		mask_type = 'A'

		network = pixelCNNModel.conv2d_layer(inputs, kernel_shape, bias_shape,
											strides, mask_type, 'conv1')

		network = pixelCNNModel.activation_fn(network, tf.nn.relu, 'conv1_act')

		# 3 Residual Blocks in the network
		for idx in xrange(self.num_residuals):
			scope = 'res' + str(idx)
			network = residual_block(network, 32, scope)

		# Final 2 Layers in the network
		kernel_shape = [1, 1, 32, 32]
		bias_shape = [32]
		strides = [1, 1, 1, 1]
		mask_type = 'B'

		for idx in xrange(2):
			scope = 'conv' + str(idx+14)
			network = pixelCNNModel.conv2d_layer(network, kernel_shape, bias_shape,
												strides, mask_type, scope)

			network = pixelCNNModel.activation_fn(network, tf.nn.relu, scope+'_act')

		# Finish up building network...

		self.network = network

	def test(self,
			inputs,
			labels,
			is_training=False):
		pass

	def train(self,
			inputs,
			labels,
			is_training=True):
		"""
			Train the Double PixelCNN model on a set of images.

			Args:
				inputs: Partial images that are learned to be completed by the model
				labels: Fully completed images for correcting error in model
				is_training: The model is in the training phase
		"""
		# input images for the model to train on
		x = tf.placeholder(tf.float32, shape=(-1, self.height, self.width,
							self.channels), name='inputs')

		# correct images for the model to check against when learning
		y = tf.placeholder(tf.float32, shape=(-1, self.height, self.width,
							self.channels), name='correct_images')

		# model is in the training phase
		is_training = tf.placeholder(tf.bool, name='training')

		# dropout rate to apply to dropout layers in model
		dropout = tf.placeholder(tf.float32, name='drop_rate')

		# build out the network architecture
		self.build_network(x, is_training, dropout)
