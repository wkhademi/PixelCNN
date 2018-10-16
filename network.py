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
		self.num_residuals = 4
		self.learning_rate = 1e-5
		self.network = None  # network graph will be built during training or testing phase
		self.loss = None # will be added to graph during training or testing phase
		self.optimizer = None # will be added to graph during training phase


	def build_network(self,
					inputs,
					labels,
					is_training,
					dropout):
		"""
			Build the 16 layer PixelCNN network.
		"""
		pixelCNNModel = PixelCNN(inputs, self.height, self.width, self.channels)

		# First layer in the network
		kernel_shape = [7, 7, self.channels, 32]
		bias_shape = [32]
		strides = [1, 1, 1, 1]
		mask_type = 'A'
		network = pixelCNNModel.conv2d_layer(inputs, kernel_shape, bias_shape,
											strides, mask_type, 'conv1')
		network = pixelCNNModel.batch_norm(network, is_training, 'conv1_batch')
		network = pixelCNNModel.activation_fn(network, tf.nn.relu, 'conv1_act')

		# 3 Residual Blocks in the network
		for idx in xrange(self.num_residuals):
			scope = 'res' + str(idx)
			network = pixelCNNModel.residual_block(network, 32, scope)

		# Final 2 Hidden Layers in the network
		kernel_shape = [1, 1, 32, 32]
		bias_shape = [32]
		strides = [1, 1, 1, 1]
		mask_type = 'B'
		for idx in xrange(2):
			scope = 'conv' + str(idx+14)
			network = pixelCNNModel.conv2d_layer(network, kernel_shape, bias_shape,
												strides, mask_type, scope)
			network = pixelCNNModel.batch_norm(network, is_training, scope+'_batch')
			network = pixelCNNModel.activation_fn(network, tf.nn.relu, scope+'_act')

		# Final Layer in the network
		kernel_shape = [1, 1, 32, self.channels]
		bias_shape = [self.channels]
		strides = [1, 1, 1, 1]
		mask_type = 'B'
		network = pixelCNNModel.conv2d_layer(inputs, kernel_shape, bias_shape,
											strides, mask_type, 'conv1')
		network = pixelCNNModel.batch_norm(network, is_training, 'conv1_batch')
		network = pixelCNNModel.activation_fn(network, tf.nn.relu, 'conv1_act')

		self.network = network

		if (is_training):
			self.loss = pixelCNNModel.loss_fn(inputs, labels, 'train_loss')
			self.optimizer = pixelCNNModel.optimizer(self.loss, self.learning_rate, 'optimizer')
		else:
			self.loss = pixelCNNModel.loss_fn(inputs, labels, 'test_loss')


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
		self.build_network(x, y, is_training, dropout)
