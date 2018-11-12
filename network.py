import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PixelCNN import PixelCNN

class Network:
	def __init__(self,
			train_inputs,
			test_inputs,
			trimmed_test_inputs,
			height,
			width,
			channels,
			config):
		self.train_inputs = train_inputs
		self.test_inputs = test_inputs
		self.trimmed_test_inputs = trimmed_test_inputs
		self.height = height
		self.width = width
		self.channels = channels
		self.config = config
		self.num_residuals = 4
		self.learning_rate = 1e-5
		self.num_epochs = 10
		self.batch_size = 32
		self.network = None  # network graph will be built during training or testing phase
		self.loss = None # will be added to graph during training or testing phase
		self.optimizer = None # will be added to graph during training phase
		self.output = None


	def build_network(self,
					inputs,
					labels,
					is_training,
					dropout,
					train):
		"""
			Build the 16 layer PixelCNN network.
		"""
		pixelCNNModel = PixelCNN(inputs, self.height, self.width, self.channels,
								self.config)

		# First layer in the network
		if (self.config == '--MNIST'):
			kernel_shape = [7, 7, self.channels, 32]
			bias_shape = [32]
		elif (self.config == '--CIFAR'):
			kernel_shape = [7, 7, self.channels, 96]
			bias_shape = [96]

		strides = [1, 1, 1, 1]
		mask_type = 'A'
		network = pixelCNNModel.conv2d_layer(inputs, kernel_shape, bias_shape,
											strides, mask_type, 'conv1')
		network = pixelCNNModel.batch_norm(network, is_training, 'conv1_batch')
		network = pixelCNNModel.activation_fn(network, tf.nn.relu, 'conv1_act')

		# 4 Residual Blocks in the network
		for idx in xrange(self.num_residuals):
			scope = 'res' + str(idx)

			if (self.config == '--MNIST'):
				network = pixelCNNModel.residual_block(network, 32, is_training, scope)
			elif (self.config == '--CIFAR'):
				network = pixelCNNModel.residual_block(network, 96, is_training, scope)

		# Final 2 Hidden Layers in the network
		if (self.config == '--MNIST'):
			kernel_shape = [1, 1, 32, 32]
			bias_shape = [32]
		elif (self.config == '--CIFAR'):
			kernel_shape = [1, 1, 96, 96]
			bias_shape = [96]

		strides = [1, 1, 1, 1]
		mask_type = 'B'
		for idx in xrange(2):
			scope = 'conv' + str(idx+14)
			network = pixelCNNModel.conv2d_layer(network, kernel_shape, bias_shape,
												strides, mask_type, scope)
			network = pixelCNNModel.batch_norm(network, is_training, scope+'_batch')
			network = pixelCNNModel.activation_fn(network, tf.nn.relu, scope+'_act')

		# Final Layer in the network
		if (self.config == '--MNIST'):
			kernel_shape = [1, 1, 32, self.channels]
		elif (self.config == '--CIFAR'):
			kernel_shape = [1, 1, 96, self.channels]

		bias_shape = [self.channels]
		strides = [1, 1, 1, 1]
		mask_type = 'B'
		network = pixelCNNModel.conv2d_layer(network, kernel_shape, bias_shape,
											strides, mask_type, 'conv16')
		network = pixelCNNModel.batch_norm(network, is_training, 'conv16_batch')

		self.network = network

		if (train == 'train'):
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

			with tf.control_dependencies(update_ops):
				self.loss = pixelCNNModel.loss_fn(network, labels, 'train_loss')
                self.output = pixelCNNModel.activation_fn(network, tf.math.sigmoid, 'train_out_act')
				self.optimizer = pixelCNNModel.optimizer(self.loss, self.learning_rate, 'optimizer')
		else:
			self.output = pixelCNNModel.activation_fn(network, tf.math.sigmoid, 'test_out_act')
			self.loss = pixelCNNModel.loss_fn(network, labels, 'test_loss')


	def generate_batch(self,
						batch_index,
						inputs):
		"""
			Generate next batch of inputs to feed into network. If batch size is
			greater than amount of inputs left, just take leftover inputs.
		"""
		num_inputs = inputs.shape[0]

		# make sure full batch of inputs can be taken from dataset
		if (self.batch_size*(batch_index+1) < num_inputs):
			image_batch = inputs[self.batch_size*batch_index:self.batch_size*(batch_index+1),:,:,:]
		else:
			image_batch = inputs[self.batch_size*batch_index:num_inputs,:,:,:]

		return image_batch


	def test(self,
			inputs,
			labels,
			training=False):
		# input images for the model to train on
		x = tf.placeholder(tf.float32, shape=(None, self.height, self.width,
							self.channels), name='inputs')

		# correct images for the model to check against when learning
		y = tf.placeholder(tf.float32, shape=(None, self.height, self.width,
							self.channels), name='correct_images')

		# model is in the training phase
		is_training = tf.placeholder(tf.bool, name='training')

		# dropout rate to apply to dropout layers in model
		dropout = tf.placeholder(tf.float32, name='drop_rate')

		# build out the network architecture
		self.build_network(x, y, is_training, dropout, 'test')

		saver = tf.train.Saver()

		with tf.Session() as sess:
			saver.restore(sess, '/tmp/model.ckpt')

			outputs, loss = sess.run([self.output, self.loss],
										feed_dict={x: inputs, y: labels,
										is_training: training, dropout: 1.0})

			fig=plt.figure(figsize=(8, 8))
			columns = 4
			rows = 5
			for i in range(1, columns*rows +1):
				img = outputs[i].reshape((28,28))
				fig.add_subplot(rows, columns, i)
				plt.imshow(img, cmap='gray')
			plt.savefig('test_output.pdf')


	def train(self,
			inputs,
			labels,
			training=True):
		"""
			Train the Double PixelCNN model on a set of images.

			Args:
				inputs: Partial images that are learned to be completed by the model
				labels: Fully completed images for correcting error in model
				is_training: The model is in the training phase
		"""
		# input images for the model to train on
		x = tf.placeholder(tf.float32, shape=(None, self.height, self.width,
							self.channels), name='inputs')

		# correct images for the model to check against when learning
		y = tf.placeholder(tf.float32, shape=(None, self.height, self.width,
							self.channels), name='correct_images')

		# model is in the training phase
		is_training = tf.placeholder(tf.bool, name='training')

		# dropout rate to apply to dropout layers in model
		dropout = tf.placeholder(tf.float32, name='drop_rate')

		# build out the network architecture
		self.build_network(x, y, is_training, dropout, 'train')

		# add ops to network to save variables
		saver = tf.train.Saver()

		# start session and train PixelCNN
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			num_batches = inputs.shape[0]/self.batch_size
			average_loss = 0

			for epoch_idx in range(self.num_epochs):
				epoch_loss = 0

				for batch_idx in range(num_batches):
					image_batch = self.generate_batch(batch_idx, inputs)

					outputs, batch_loss, _ = sess.run([self.output, self.loss, self.optimizer],
												feed_dict={x: image_batch, y: image_batch,
												is_training: training, dropout: 0.2})

					epoch_loss += batch_loss

				average_loss = epoch_loss / num_batches
				print('Average Loss: ', average_loss, ' for epoch ', epoch_idx+1)

            fig=plt.figure(figsize=(8, 8))
			columns = 4
			rows = 5
			for i in range(1, columns*rows +1):
				img = outputs[i].reshape((28,28))
				fig.add_subplot(rows, columns, i)
				plt.imshow(img, cmap='gray')
			plt.savefig('train_output.pdf')

			# save model to disk
			save_path = saver.save(sess, '/tmp/model.ckpt')
