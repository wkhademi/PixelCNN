import tensorflow as tf
import numpy as np


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
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable(name='weights', shape=kernel_shape, dtype=tf.float32,
                                    initializer=tf.glorot_uniform_initializer())

            if (mask_type != None):  # use convolution mask
                mask = np.ones(kernel_shape).astype('f')

                # find center of weight mask matrix
                h_center =  kernel_shape[0] // 2
                w_center = kernel_shape[1] // 2

                # zero everything that comes after center pixel in weight mask matrix
                mask[h_center+1:, :, :, :] = 0.0
                mask[h_center, w_center+1:, :, :] = 0.0

                # tie channel inputs to each other and itself
                # Note: blue channel retains all center pixel information
                if (self.config == '--CIFAR'):
                    red_filter_end = kernel_shape[3]/3
                    mask[h_center, w_center, 1:, :red_filter_end] = 0.0

                    green_filter_end = 2*red_filter_end
                    mask[h_center, w_center, 2:, red_filter_end:green_filter_end] = 0.0

                # remove center pixel information availability
                if (mask_type == 'A'):
                    if (self.config == '--MNIST' or self.config == '--FREY'):
                        mask[h_center, w_center, :, :] = 0.0
                    elif (self.config == '--CIFAR'): # tie channel inputs to each other
                        mask[h_center, w_center, :, :red_filter_end] = 0.0
                        mask[h_center, w_center, 1:, red_filter_end:green_filter_end] = 0.0
                        blue_filter_end = kernel_shape[3]
                        mask[h_center, w_center, 2:, green_filter_end:blue_filter_end] = 0.0

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
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
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
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # downsample features from num_features -> 0.5*num_features
            kernel_shape = [1, 1, features, features/2]
            bias_shape = [features/2]
            strides = [1, 1, 1, 1]
            conv1 = self.conv2d_layer(inputs, kernel_shape, bias_shape, strides, mask_type, 'res_conv1')
            conv1_norm = self.batch_norm(conv1, is_training, 'res_batch1')
            act1 = self.activation_fn(conv1_norm, tf.nn.relu, 'res_act1')

            # convolution layer
            kernel_shape = [7, 7, features/2, features/2]
            bias_shape = [features/2]
            conv2 = self.conv2d_layer(act1, kernel_shape, bias_shape, strides, mask_type, 'res_conv2')
            conv2_norm = self.batch_norm(conv2, is_training, 'res_batch2')
            act2 = self.activation_fn(conv2_norm, tf.nn.relu, 'res_act2')

            # upsample features from 0.5*num_features -> num_features
            kernel_shape = [1, 1, features/2, features]
            bias_shape = [features]
            conv3 = self.conv2d_layer(act2, kernel_shape, bias_shape, strides, mask_type, 'res_conv3')
            conv3_norm = self.batch_norm(conv3, is_training, 'res_batch3')
            act3 = self.activation_fn(conv3_norm, tf.nn.relu, 'res_act3')

            # add input map to output map
            res_output = inputs + act3

        return res_output


    def batch_norm(self,
                    inputs,
                    is_training,
                    scope):
        """
            Performs batch normalization on the input layer.
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            batch_norm = tf.layers.batch_normalization(inputs, training=is_training, name='batch_norm')

        return batch_norm


    def dropout(self,
                inputs,
                drop_rate,
                is_training,
                scope):
        """
            Performs dropout on the input layer.
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            drop = tf.layers.dropout(inputs, rate=drop_rate, training=is_training, name='dropout')

        return drop


    def loss_fn(self,
                inputs,
                labels,
                scope):
        """
            Calculates the average loss of all the predicted pixels in an image
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if (self.config == '--MNIST'):
                cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=inputs, name='loss')
            elif (self.config == '--CIFAR' or self.config == '--FREY'):
                labels = tf.cast(labels, tf.int32)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=inputs, name='loss')

            loss = tf.reduce_mean(cross_entropy)

        return loss


    def optimizer(self,
                    loss,
                    learning_rate,
                    scope):
        """
            Update weights and biases of network to minimize loss
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        return optimize


    def flatten(self,
                inputs,
                scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            shape = inputs.get_shape().as_list()
            new_shape = [-1, shape[1]*shape[2]*shape[3]]
            flat = tf.reshape(inputs, new_shape)

        return flat
