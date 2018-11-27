import tensorflow as tf
import numpy as np
from PixelCNN import PixelCNN
from utils import trim_images, save_samples, binarize

class Network:
	def __init__(self,
			train_inputs,
			test_inputs,
			height,
			width,
			channels,
			config):
		self.train_inputs = train_inputs
		self.test_inputs = test_inputs
		self.height = height
		self.width = width
		self.channels = channels
		self.config = config
		self.num_residuals = 3
		self.learning_rate = 1e-4
		self.num_epochs = 25
		self.batch_size = 16
		self.network = None  # network graph will be built during training or testing phase
		self.loss = None # will be added to graph during training or testing phase
		self.optimizer = None # will be added to graph during training phase
		self.pred = None


	def build_network(self,
					inputs,
					labels,
					is_training,
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
			kernel_shape = [7, 7, self.channels, 256]
			bias_shape = [256]

		strides = [1, 1, 1, 1]
		mask_type = 'A'
		network = pixelCNNModel.conv2d_layer(inputs, kernel_shape, bias_shape, strides, mask_type, 'conv1')
		network = pixelCNNModel.batch_norm(network, is_training, 'conv1_norm')

		# 4 Residual Blocks in the network
		for idx in xrange(self.num_residuals):
			scope = 'res' + str(idx)
			if (self.config == '--MNIST'):
				network = pixelCNNModel.residual_block(network, 32, is_training, scope)
			elif (self.config == '--CIFAR'):
				network = pixelCNNModel.residual_block(network, 256, is_training, scope)

		if (self.config == '--MNIST'):
			kernel_shape = [1, 1, 32, 32]
			bias_shape = [32]
		elif (self.config == '--CIFAR'):
			kernel_shape = [1, 1, 256, 1024]
			bias_shape = [1024]

		network = pixelCNNModel.conv2d_layer(network, kernel_shape, bias_shape, strides, mask_type, 'conv11')
		network = pixelCNNModel.batch_norm(network, is_training, 'conv11_norm')

		if (self.config == '--CIFAR'):
			kernel_shape = [1, 1, 1024, 1024]

		strides = [1, 1, 1, 1]
		mask_type = 'B'
		network = pixelCNNModel.activation_fn(network, tf.nn.relu, 'act12')
		network = pixelCNNModel.conv2d_layer(network, kernel_shape, bias_shape, strides, mask_type, 'conv12')
		network = pixelCNNModel.batch_norm(network, is_training, 'conv12_norm')

		# Final Layer in the network
		if (self.config == '--MNIST'):
			kernel_shape = [1, 1, 32, self.channels]
			bias_shape = [self.channels]
		elif (self.config == '--CIFAR'):
			kernel_shape = [1, 1, 1024, 256*self.channels]
			bias_shape = [256*self.channels]

		strides = [1, 1, 1, 1]
		mask_type = 'B'
		network = pixelCNNModel.activation_fn(network, tf.nn.relu, 'act13')
		network = pixelCNNModel.conv2d_layer(network, kernel_shape, bias_shape, strides, mask_type, 'conv13')
		network = pixelCNNModel.batch_norm(network, is_training, 'conv13_norm')

		self.network = network

		if (train == 'train'):
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

			with tf.control_dependencies(update_ops):
				self.loss = pixelCNNModel.loss_fn(network, labels, 'train_loss')
				self.optimizer = pixelCNNModel.optimizer(self.loss, self.learning_rate, 'optimizer')
		else:
			self.loss = pixelCNNModel.loss_fn(network, labels, 'test_loss')

			if (self.config == '--MNIST'):
				self.pred = pixelCNNModel.activation_fn(network, tf.math.sigmoid, 'test_out_act')
			elif (self.config == 'CIFAR'):
				pass # Do something else...


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
			training=False):
		# input images for the model to train on
		x = tf.placeholder(tf.float32, shape=(None, self.height, self.width,
							self.channels), name='inputs')

		# correct images for the model to check against when learning
		y = tf.placeholder(tf.float32, shape=(None, self.height, self.width,
							self.channels), name='correct_images')

		# model is in the training phase
		is_training = tf.placeholder(tf.bool, name='training')

		# build out the network architecture
		self.build_network(x, y, is_training, 'test')

		saver = tf.train.Saver()

		with tf.Session() as sess:
			saver.restore(sess, '/tmp/model.ckpt')

			# remove bottom half from images
			images = trim_images(self.test_inputs)

			# use model to generate bottom half of images
			for i in range(self.height//2, self.height):
				for j in range(self.width):
					for k in range(self.channels):
						probs = sess.run(self.pred, feed_dict={x: images, y: self.test_inputs, is_training: training})
						sample = binarize(probs)
						images[:, i, j, k] = sample[:, i, j, k]

			save_samples(images, self.height, self.width)


	def train(self,
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

		# build out the network architecture
		self.build_network(x, y, is_training, 'train')

		# add ops to network to save variables
		saver = tf.train.Saver()

		# start session and train PixelCNN
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			num_batches = self.train_inputs.shape[0]/self.batch_size
			average_loss = 0

			for epoch_idx in range(self.num_epochs):
				epoch_loss = 0

				# shuffle inputs every epoch
				np.random.shuffle(self.train_inputs)

				for batch_idx in range(num_batches):
					image_batch = self.generate_batch(batch_idx, self.train_inputs)

					batch_loss, _ = sess.run([self.loss, self.optimizer],
											feed_dict={x: image_batch, y: image_batch, is_training: training})

					epoch_loss += batch_loss

				average_loss = epoch_loss / num_batches
				print('Average Loss: ', average_loss, ' for epoch ', epoch_idx+1)

			# save model to disk
			save_path = saver.save(sess, '/tmp/model.ckpt')
