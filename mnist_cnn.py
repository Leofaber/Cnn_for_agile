"""
Esempio: CIFAR-10

        32x32 , colored images, 50.000 images for training
                                10.000 images for test
        10 classes, 6.000 images per class

"""

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# HELPER

# INIT WEIGHTS
def init_weights(shape):
    init_random_distribution = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_distribution)

# INIT BIAS
def init_bias(shape):
    #print "@_@ ", shape
    init_bias_vals = tf.constant(0.1, shape=[shape])
    return tf.Variable(init_bias_vals)

# CONV2D
def conv2d(x, W):
    # x --> input tensor [batch, H, W, Channels IN]
    # W --> [filter H, filter W, Channels IN, Channels OUT]
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# POOLING
def max_pool_2by2(x):
    # x --> [batch,H,W,Channles] :
    #       bunch of images, height and width and channels for each image
    # ksize --> size of the windows for each dimension of input tensor
    #       [1,2,2,1] pooling on H and W of each individual image (2x2 pooling)
    # strides --> stride of the sliding window for each dimension
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# Create a CONVOLUTIONAL layer
def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias(shape[3])
    return tf.nn.relu(conv2d(input_x,W)+b)

# Create a FULLY CONNECTED layer
def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias(size)
    return tf.matmul(input_layer, W) + b


# Placeholders
# shape = [dont know how many, number of pixels]
x = tf.placeholder(tf.float32, shape=[None, 784])

# Labels
y_true = tf.placeholder(tf.float32, shape=[None,10]) # 0,..,9

# LAYERS
#  -1 for batch size, which specifies that this dimension should be
# dynamically computed based on the number of input values in
# features["x"], holding the size of all other dimensions constant.
# This allows us to treat batch_size as a hyperparameter that we can
# tune.
# Our MNIST dataset is composed of monochrome 28x28 pixel images,
# so the desired shape for our input layer is [batch_size, 28, 28, 1].


x_image = tf.reshape(x, [-1,28,28,1])

# shape:
# 5x5 = kernel size
# 1 = greyscale image
# 32 features for each 5x5 patch (number of output channels)
convo_1 = convolutional_layer(x_image, shape=[5,5,1,32])
convo_1_pooling = max_pool_2by2(convo_1)


convo_2 = convolutional_layer(convo_1_pooling, shape=[5,5,32,64])
convo_2_pooling = max_pool_2by2(convo_2)

# We've got to flat the output in order to connect it with a
# fully connected layer
# The previous output was 64 ->
convo_2_flat = tf.reshape(convo_2_pooling, [-1, 7*7*64])

# 1024 -> number of neurons
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))


# DROP OUT
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

# 10 -> classes
y_pred = normal_full_layer(full_one_dropout, 10)


# Loss function
average = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
cross_entropy = tf.reduce_mean(average)

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

# Run Session
steps = 50000

with tf.Session() as sess:
    sess.run(init)

    for i in range(steps):

        batch_x, batch_y = mnist.train.next_batch(10)
        #batch_x.expand
        #print "O_O ", batch_x
        #print "X_X ", batch_y
        sess.run(train, feed_dict={x:batch_x, y_true:batch_y,hold_prob:0.5})

        #print "G_G ", x_image

        if i%100 == 0:
            print "\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
            print("ON STEP: {}".format(i))

            print("ACCURACY: ")
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))
            print(sess.run(acc,feed_dict={x:mnist.test.images, y_true:mnist.test.labels,hold_prob:1.0}))
            print('\n')



            print mnist.test.images.shape
            print mnist.test.labels.shape
            """
            print("Average",sess.run(average,                           feed_dict={x:mnist.test.images, y_true:mnist.test.labels, hold_prob:1.0}))
            print("cross_entropy",sess.run(cross_entropy,               feed_dict={x:mnist.test.images, y_true:mnist.test.labels, hold_prob:1.0}))
            """
            #print y_pred.shape
            #print "y_pred",sess.run(y_pred,                           feed_dict={x:mnist.test.images, y_true:mnist.test.labels, hold_prob:1.0})
            #print "average", sess.run(average, feed_dict={x:mnist.test.images, y_true:mnist.test.labels, hold_prob:1.0})
            _y_pred = sess.run(y_pred,                           feed_dict={x:mnist.test.images, y_true:mnist.test.labels, hold_prob:1.0})
            #sum = 0
            #for i in average:
            #        sum += i
            #print "SUM: ", sum
            """
            print "y_true",sess.run(y_true,                             feed_dict={x:mnist.test.images, y_true:mnist.test.labels, hold_prob:1.0})
            print "tf.argmax(y_pred,1)",sess.run(tf.argmax(y_pred,1),   feed_dict={x:mnist.test.images, y_true:mnist.test.labels, hold_prob:1.0})
            print "tf.argmax(y_true,1)",sess.run(tf.argmax(y_pred,1),   feed_dict={x:mnist.test.images, y_true:mnist.test.labels, hold_prob:1.0})
            """
