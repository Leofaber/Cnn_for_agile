import tensorflow as tf
from agile_batch_data_handler import Dataset
from utility import *

"""
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
         L O A D I N G     D A T A S E T
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""
agile_dataset = Dataset('AGILE DATASET', 500, 'training_set/', 'test_set/')
agile_dataset.check_load()

batch_tr = agile_dataset.get_batch('TRAIN')

print_2D_matrix_on_window(batch_tr[0][0])
"""
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                HELPER FUNCTIONS
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""

def init_weights(shape):
    init_random_distribution = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_distribution)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=[shape])
    return tf.Variable(init_bias_vals)

# x --> input tensor of shape [batch, in_height, in_width, in_channels]
# W --> filter / kernel tensor of shape [filter_height, filter_width,
#       in_channels, out_channels]
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# x ->  A 4-D Tensor with shape [batch, height, width, channels] and type tf.float32
# ksize ->  A list of ints that has length >= 4. The size of the window for each
#           dimension of the input tensor.
def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


""" Create convolutional_layer """
def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias(shape[3])
    return tf.nn.relu(conv2d(input_x,W)+b)

# NORMAL (FULLy CONNECTED LAYER)
def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias(size)
    return tf.matmul(input_layer, W) + b
