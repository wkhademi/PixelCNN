import tensorflow as tf
import numpy as np
from PixelCNN import PixelCNN
from utils import trim_images, save_samples, binarize, sample_categorical

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
        self.num_residuals = 7
        self.learning_rate = 1e-3
        self.num_epochs = 50
        self.batch_size = 64
        self.network = None
        self.loss = None
        self.optimizer = None
        self.pred = None
        self.NLL = None


    def build_network(self, inputs, labels, is_training, train):
        pixelCNNModel = PixelCNN(inputs, self.height, self.width, self.channels, self.config)

        # First layer in the network
        if (self.config == '--MNIST'):
            kernel_shape = [7, 7, self.channels, 64]
            bias_shape = [64]
        elif (self.config == '--CIFAR'):
            kernel_shape = [7, 7, self.channels, 256]
            bias_shape = [256]
        elif (self.config == '--FREY'):
            kernel_shape = [7, 7, self.channels, 128]
            bias_shape = [128]

        strides = [1, 1, 1, 1]
        mask_type = 'A'
        network = pixelCNNModel.conv2d_layer(inputs, kernel_shape, bias_shape, strides, mask_type, 'conv1')
        network = pixelCNNModel.batch_norm(network, is_training, 'conv1_norm')
        network = pixelCNNModel.activation_fn(network, tf.nn.relu, 'act1')

        if (self.config == '--MNIST'):
            kernel_shape = [7, 7, 64, 64]
        elif (self.config == '--CIFAR'):
            kernel_shape = [7, 7, 256, 256]
        elif (self.config == '--FREY'):
            kernel_shape = [7, 7, 128, 128]
            mask_type = 'B'

        # 7 Mask Type 'B' Convolutions
        for idx in xrange(self.num_residuals):
            scope = 'res' + str(idx)
            network = pixelCNNModel.conv2d_layer(network, kernel_shape, bias_shape, strides, mask_type, scope+'conv')
            network = pixelCNNModel.batch_norm(network, is_training, scope+'conv_norm')
            network = pixelCNNModel.activation_fn(network, tf.nn.relu, scope+'act')

        # Final Layer in the network
        if (self.config == '--MNIST'):
            kernel_shape = [1, 1, 64, self.channels]
            bias_shape = [self.channels]
        elif (self.config == '--CIFAR'):
            kernel_shape = [1, 1, 256, 256*self.channels]
            bias_shape = [256*self.channels]
        elif (self.config == '--FREY'):
            kernel_shape = [1, 1, 128, 256*self.channels]
            bias_shape = [256*self.channels]

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
                self.NLL = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=network))
                self.pred = pixelCNNModel.activation_fn(network, tf.nn.sigmoid, 'test_out_act')
            elif (self.config == 'CIFAR' or self.config == '--FREY'):
                self.NLL = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.cast(labels, tf.int32), logits=network))
                self.pred = tf.cast(tf.argmax(pixelCNNModel.activation_fn(network, tf.nn.softmax, 'test_out_act'), axis=-1), tf.int8)


    def generate_batch(self, batch_index, inputs):
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


    def test(self, training=False):
        # input images for the model to train on
        x = tf.placeholder(tf.float32, shape=(None, self.height, self.width, self.channels), name='inputs')

        # correct images for the model to check against when learning
        if (self.config == '--MNIST'):
            y = tf.placeholder(tf.float32, shape=(None, self.height, self.width, self.channels), name='correct_images')
        elif (self.config == '--CIFAR' or self.config == '--FREY'):
            y = tf.placeholder(tf.float32, shape=(None, self.height, self.width, 256*self.channels), name='correct_images')

        # model is in the training phase
        is_training = tf.placeholder(tf.bool, name='training')

        # build out the network architecture
        self.build_network(x, y, is_training, 'test')

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, '/tmp/model.ckpt')

            loss = []
            NLL = []
            for i in range(self.test_inputs.shape[0]):
                if (self.config == '--MNIST'):
                    binaryImage = binarize(self.test_inputs[i])
                    binaryImage = np.reshape(binaryImage, (1, 28, 28, 1))
                    loss.append(sess.run(self.loss, feed_dict={x: binaryImage, y: binaryImage, is_training: training}))
                    NLL.append(sess.run(self.NLL, feed_dict={x: binaryImage, y: binaryImage, is_training: training}))
                elif (self.config == '--FREY'):
                    image = self.test_inputs[i]
                    label = sess.run(tf.one_hot(np.reshape(image, (-1, 28, 20)), 256, axis=-1))
                    image = 2. * (np.reshape(image, (1, 28, 20, 1)).astype('float32') / 255.) - 1.
                    loss.append(sess.run(self.loss, feed_dict={x: image, y: label, is_training: training}))
                    NLL.append(sess.run(self.NLL, feed_dict={x: image, y: label, is_training: training}))

            loss = np.mean(loss)
            NLL = np.mean(NLL)
            print("Test loss: ", loss, "NLL: ", NLL)

            #save_samples(self.test_inputs[:22], self.height, self.width)

            # remove bottom half from images
            images = trim_images(self.test_inputs)

            if (self.config == '--MNIST'):
                images = images[:22]
                self.test_inputs = self.test_inputs[:22]
            elif (self.config == '--CIFAR' or self.config == '--FREY'):
                images = 2 * (images[:22].astype('float32') / 255.) - 1.
                self.test_inputs = sess.run(tf.one_hot(np.reshape(self.test_inputs[:22], (-1, 28, 20)), 256, axis=-1))

            #save_samples(images, self.height, self.width)

            # use model to generate bottom half of images
            for i in range(self.height//2, self.height):
                for j in range(self.width):
                    for k in range(self.channels):
                        probs = sess.run(self.pred, feed_dict={x: images, y: self.test_inputs, is_training: training})

                        if (self.config == '--MNIST'):
                            sample = binarize(probs)
                        elif (self.config == '--FREY'):
                            sample = sample_categorical(probs[:, i, j])

                        images[:, i, j, k] = sample[:, i, j]

        save_samples(images, self.height, self.width)


    def train(self, training=True):
        """
            Train the Double PixelCNN model on a set of images.

            Args:
                inputs: Partial images that are learned to be completed by the model
                labels: Fully completed images for correcting error in model
                is_training: The model is in the training phase
        """
        # input images for the model to train on
        x = tf.placeholder(tf.float32, shape=(None, self.height, self.width, self.channels), name='inputs')

        # correct images for the model to check against when learning
        if (self.config == '--MNIST'):
            y = tf.placeholder(tf.float32, shape=(None, self.height, self.width, self.channels), name='correct_images')
        elif(self.config == '--CIFAR' or self.config == '--FREY'):
            y = tf.placeholder(tf.float32, shape=(None, self.height, self.width, 256*self.channels), name='correct_images')

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

                    if (self.config == '--MNIST'):
                        image_batch = binarize(image_batch)
                        batch_loss, _ = sess.run([self.loss, self.optimizer], feed_dict={x: image_batch, y: image_batch, is_training: training})
                    elif (self.config == '--CIFAR' or self.config == '--FREY'):
                        images = 2. * (image_batch.astype('float32') / 255.) - 1.
                        labels = sess.run(tf.one_hot(np.reshape(image_batch, (-1, 28, 20)), 256, axis=-1))
                        batch_loss, _ = sess.run([self.loss, self.optimizer], feed_dict={x: images, y: labels, is_training: training})

                    epoch_loss += batch_loss

                average_loss = epoch_loss / num_batches
                print('Average Loss: ', average_loss, ' for epoch ', epoch_idx+1)

        # save model to disk
        save_path = saver.save(sess, '/tmp/model.ckpt')
