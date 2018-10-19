import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class PixelCNN:
	"""
		PixelCNN Model Layers and functions needed to train (i.e. loss function, optimizer, etc.)
	"""
	def __init__(self,
				inputs,
				height,
				width,
				channels,
				config):
		self.inputs = inputs
		self.height = height
		self.width = width
		self.channels = channels
		self.config = config


	def conv2d_layer(self,
					inputs,
					kernel_shape,
					bias_shape,
					strides,
					mask_type,
					scope):
		"""
			Convolves each input with a set of filters.

			Args:
				inputs:
				kernel_shape:
				bias_shape:
				strides:
				mask_type:
				scope:

			Returns:
				pre_activation:
		"""
		with tf.variable_scope(scope):
			weights = tf.get_variable(name='weights', shape=kernel_shape, dtype=tf.float32,
									initializer=tf.glorot_uniform_inializer())

			if mask_type != None:  # use convolution mask
				mask = np.ones(kernel_shape).astype('f')

				# find center of weight mask matrix
				h_center =  kernel_shape[0] // 2
				w_center = kernel_shape[1] // 2

				# zero everything that comes after center pixel in weight mask matrix
				mask[h_center+1:, :, :, :] = 0.0
				mask[h_center, w_center+1, :, :] = 0.0

				if mask_type == 'a':  # zero out center pixel
					mask[h_center, w_center:, :, :] = 0.0

				# update weights with mask
				weights = tf.multiply(weights, tf.constant(mask, dtype=tf.float32))
				tf.add_to_collections('weights_{}'.format(mask_type), weights)

			# perform convolutions on image batch
			conv2d = tf.nn.conv2d(inputs, weights, strides, padding='SAME', name='conv2d')

			# adjust output with a bias term
			biases = tf.get_variable(name='biases', shape=bias_shape, dtype=tf.float32,
									initializer=tf.constant_initializer(0.0))

			pre_activation = tf.nn.bias_add(conv2d, biases, name='conv2d_preact')

		return pre_activation


	def activation_fn(self,
					inputs,
					fn,
					scope):
		"""
			Applies non-linear activation function to layer.

			Args:
				inputs: batch of images
				fn: non-linear activation function (i.e. ReLu, Leaky ReLU, tanh, etc.)
				scope: activation layer scope

			Returns:
				activation: A layer in which a non-linear activation has been applied.
		"""
		with tf.variable_scope(scope):
			activation = fn(inputs, name='activation')

		return activation


	def residual_block(self,
					inputs,
					features,
					is_training,
					scope,
					mask_type='B'):
		"""
			Residual Block for PixelCNN architecture.

			Args:
				inputs: batch of images
				features: num features in the input and output layer
				is_training: whether model is in training phase or not
				scope: residual block scope
				mask_type: weight mask type for convolution layers

			Returns:
				res_output: the input map added to the output map of the residual block
		"""
		with tf.variable_scope(scope):
			# downsample features from num_features -> 0.5*num_features
			kernel_shape = [1, 1, features, features/2]
			bias_shape = [features/2]
			strides = [1, 1, 1, 1]
			conv1 = self.conv2d_layer(inputs, kernel_shape, bias_shape, strides, mask_type, 'res_conv1')
			conv1_batch = self.batch_norm(conv1, is_training, 'res_batch1')
			conv1_out = self.activation_fn(conv1_batch, tf.nn.relu, 'res_act1')

			# convolution layer
			kernel_shape = [3, 3, features/2, features/2]
			bias_shape = [features/2]
			conv2 = self.conv2d_layer(conv1_out, kernel_shape, bias_shape, strides, mask_type, 'res_conv2')
			conv2_batch = self.batch_norm(conv2, is_training, 'res_batch2')
			conv2_out = self.activation_fn(conv2_batch, tf.nn.relu, 'res_act2')

			# upsample features from num_features -> 2*num_features
			kernel_shape = [1, 1, features/2, features]
			bias_shape = [features]
			conv3 = self.conv2d_layer(conv2_out, kernel_shape, bias_shape, strides, mask_type, 'res_conv3')
			conv3_batch = self.batch_norm(conv3, is_training, 'res_batch3')
			conv3_out = self.activation_fn(conv3_batch, tf.nn.relu, 'res_act3')

			# add input map to output map
			res_output = inputs + conv3_out

		return res_output


	def batch_norm(self,
				inputs,
				is_training,
				scope):
		"""
			Performs batch normalization on the input layer.
		"""
		with tf.variable_scope(scope):
			batch_norm = tf.layers.batch_normalization(inputs, is_training=is_training,
													   name='batch_norm')

		return batch_norm


	def dropout(self,
				inputs
				drop_rate,
				is_training,
				scope):
		"""
			Performs dropout on the input layer.
		"""
		with tf.variable_scope(scope):
			drop = tf.layers.dropout(inputs, rate=drop_rate, training=is_training,
									 name='dropout')


	def loss_fn(self,
				inputs,
				labels,
				scope):
		"""
			Calculates the average loss of all the predicted pixels in an image
		"""
		with tf.variable_scope(scope):
			if (self.config == '--MNIST'):
				loss_distribution = tf.nn.sigmoid_cross_entropy_with_logits(logits=inputs,
																labels=labels, name='loss')
			elif (self.config == '--CIFAR'):
				loss_distribution = tf.nn.softmax_cross_entropy_with_logits_v2(logits=inputs,
																labels=labels, name='loss')

			loss = tf.reduce_mean(loss_distribution)

		return loss


	def optimizer(self,
				loss,
				learning_rate
				scope):
		"""
			Update weights and biases of network to minimize loss
		"""
		with tf.variable_scope(scope):
			optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

		return optimize
