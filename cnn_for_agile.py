import tensorflow as tf
from agile_batch_data_handler import Dataset
from utility import *

title = """
            @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
             C O N V O L U T I O N A L   N E U R A L   N E T W O R K   F O R   A G I L E
            @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        """

print title

"""
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
         L O A D I N G     D A T A S E T
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""
agile_dataset = Dataset('AGILE DATASET', 'batch_smoothed_dataset/training_set/', 'batch_smoothed_dataset/test_set/')
agile_dataset.check_load()
total_training_samples_number = agile_dataset.get_number_of_training_samples()
print "\n*Total training samples number: ",total_training_samples_number

"""
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                P A R A M E T E R S
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""

# Architecture
conv_kernel_size = 7  #(2 * PSF/CDELT2) + 1;
number_of_filters = 32
neurons_number_classification_layer = 1024
# Training
batch_size = 50
learning_rate = 0.0001
hold_probability = 0.5 # during dropout

steps = total_training_samples_number/batch_size # I give all the training examples
print "*Number of steps : ",steps






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

"""
    plot_image_with_filters(1,batch_x[1], 1, features , cmap_modes = ['gist_gray', 'hot'])
"""




"""
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            P L A C E H O L D E R S
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""

# x is the placeholder for the input data. Depending on batch size, it may have
# a different shape. So I put 'None'  as first parameter of the shape object.

# x --> input tensor of shape [batch, in_height, in_width, in_channels]
input_x = tf.placeholder(tf.float32, shape=[None, 96, 96, 1])

# y_true is the placeholder for the labels. Depending on batch size, it may have
# a different shape. So I put 'None'  as first parameter of the shape object.
# The second parameter is the number of classes in this case is 2 (0 and 1)
y_true = tf.placeholder(tf.float32, shape=[None,2])
#batch_tr = agile_dataset.get_batch('TRAIN')
#print_2D_matrix_on_window(batch_tr[0][0])


"""
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            T H E    N E T W O R K
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""



""" Create convolutional layer (1) """
# filter tensor of shape [filter_height, filter_width, in_channels, out_channels]
# Il kernel del filtro e' 5x5 , viene applicato ad 1 sola immagine e abbiamo 32
# diversi filtri.
kernels_layers = [conv_kernel_size,conv_kernel_size,1,number_of_filters]

W = init_weights(kernels_layers)
b = init_bias(kernels_layers[3]) # bias in applied to each neuron

# La convoluzione viene effettuata con stride = 1 in ogni dimensione,
# padding = SAME significa che l'output avra' la stessa dimensione dell'input (ma una dimensione in piu' (depth) perche' ci sono 32 filtri)
convo_1 = tf.nn.conv2d(input_x, W, strides=[1,1,1,1], padding='SAME') + b
print 'convo_1',convo_1.shape
""" -------  ------- """

""" Create a RELU layer (1) """
convo_1_relu = tf.nn.relu(convo_1)
print 'convo_1_relu',convo_1_relu.shape
""" -----------(?, 96, 96, 16) ------------------------ """



""" Create convolutional layer (2) """

kernels_layers = [conv_kernel_size,conv_kernel_size,number_of_filters,number_of_filters]# """ l'output di convo_1_relu sono 16 canali! """

W = init_weights(kernels_layers)
b = init_bias(kernels_layers[3]) # bias in applied to each neuron

convo_2 = tf.nn.conv2d(convo_1_relu, W, strides=[1,1,1,1], padding='SAME') + b

print 'convo_2',convo_2.shape
""" -------   ------- """


""" Create a RELU layer (2) """
convo_2_relu = tf.nn.relu(convo_2)
print 'convo_2_relu',convo_2_relu.shape
""" ----------------------------------- """





""" Create a POOLING layer (1) """
# ksize = Il pooling viene fatto su singole immagini , e' un pooling 2x2 e sull'unico canale (gray scale)
convo_1_pooling = tf.nn.max_pool(convo_2_relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
print 'convo_1_pooling',convo_1_pooling.shape
""" ------- (?, 42, 42, 16) ------- """






""" Create convolutional layer (3) """

kernels_layers = [conv_kernel_size,conv_kernel_size,number_of_filters,number_of_filters]

W = init_weights(kernels_layers)
b = init_bias(kernels_layers[3]) # bias in applied to each neuron


convo_3 = tf.nn.conv2d(convo_1_pooling, W, strides=[1,1,1,1], padding='SAME') + b
print 'convo_3',convo_3.shape
""" -------  ------- """

""" Create a RELU layer (3) """
convo_3_relu = tf.nn.relu(convo_3)
print 'convo_3_relu',convo_3_relu.shape
""" -----------(?, 45, 45, 16) ------------------------ """



""" Create convolutional layer (4) """

kernels_layers = [conv_kernel_size,conv_kernel_size,number_of_filters,number_of_filters]# """ l'output di convo_1_relu sono 16 canali! """

W = init_weights(kernels_layers)
b = init_bias(kernels_layers[3]) # bias in applied to each neuron

convo_4 = tf.nn.conv2d(convo_3_relu, W, strides=[1,1,1,1], padding='VALID') + b

print 'convo_4',convo_4.shape
""" -------   ------- """


""" Create a RELU layer (4) """
convo_4_relu = tf.nn.relu(convo_4)
print 'convo_4_relu',convo_4_relu.shape
""" ------------39, 39, 16 ----------------------- """




""" Create a POOLING layer (2) """
# ksize = Il pooling viene fatto su singole immagini , e' un pooling 2x2 e sull'unico canale (gray scale)
convo_2_pooling = tf.nn.max_pool(convo_4_relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print 'convo_2_pooling',convo_2_pooling.shape
""" -------  ------- """





""" ATTACH HERE THE LAST LAYER OUTPUT """
last_layer_neurons = int(convo_2_pooling.shape[1]*convo_2_pooling.shape[2]*convo_2_pooling.shape[3])
""" --------------------------------- """

""" Create a FULLY CONNECTED layer (1) - WITH DROPOUT """
convo_to_flat = tf.reshape(convo_2_pooling, [-1, last_layer_neurons])
# (?, 21904)

input_size = int(convo_to_flat.get_shape()[1])
W = init_weights([input_size, neurons_number_classification_layer])
b = init_bias(neurons_number_classification_layer)
full_1 = tf.nn.relu(tf.matmul(convo_to_flat, W) + b)
print "full_1",full_1.shape
# (?, 1024)

hold_prob = tf.placeholder(tf.float32)
full_1_dropout = tf.nn.dropout(full_1, keep_prob=hold_prob)
# (?, 1024)
""" ----------------------------------- """

""" Create a FULLY CONNECTED layer (2) """
input_size = int(full_1_dropout.get_shape()[1])
W = init_weights([input_size, 2]) # 2 classes
b = init_bias(2)
y_pred = tf.nn.relu(tf.matmul(full_1_dropout, W) + b)
# (?, 2)
""" ----------------------------------- """


"""
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        L O S S   F U N C T I O N   A N D   O P T I M I Z E R
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""
# Loss function
average = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
cross_entropy = tf.reduce_mean(average)

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cross_entropy)

"""
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            T H E  D E B U G  S E S S I O N
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""
raw_input("\n** Press any key to start! ")

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    """ ADD EPOCHs """
    """ Randomize traininig set """

    for i in range(steps):

        batch_x, batch_y = agile_dataset.get_batch(50,'TRAIN')

        sess.run(train, feed_dict={ input_x : batch_x, y_true : batch_y, hold_prob:hold_probability})


        """ Compute ACCURACY """
        if i%100 == 0: # after 100 * batch_size examples
            print "\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
            print("ON STEP: {}".format(i))
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))
            test_batch_x, test_batch_y = agile_dataset.get_batch(1000,'TEST')

            print("Accuracy: ")
            #print(sess.run(acc,feed_dict={ input_x :test_batch_x, y_true:test_batch_y,hold_prob:1.0}))
            #print sess.run(y_pred,feed_dict={ input_x :test_batch_x, y_true:test_batch_y,hold_prob:1.0})
            #print sess.run(tf.argmax(y_pred,1),feed_dict={ input_x :test_batch_x, y_true:test_batch_y,hold_prob:1.0})
            #print sess.run(y_true,feed_dict={ input_x :test_batch_x, y_true:test_batch_y,hold_prob:1.0})
            print('\n')

        """ Show filters """
        if i%1000 == 0: # after 1000 * batch_size examples
            #plot_image(0, batch_x[0], 'hot')
            plot_filters(1, sess.run(convo_2_relu,feed_dict={ input_x :test_batch_x, y_true:test_batch_y,hold_prob:0.5}), 0, 'hot')
            raw_input()


        #print sess.run(y_pred,feed_dict={ input_x :test_batch_x, y_true:test_batch_y,hold_prob:1.0})
        #print sess.run(y_true,feed_dict={ input_x :test_batch_x, y_true:test_batch_y,hold_prob:1.0})
        #print sess.run(matches,feed_dict={ input_x :test_batch_x, y_true:test_batch_y,hold_prob:1.0})





    #conv1 = sess.run(convo_1, feed_dict={input_x:batch_x})
    #print conv1.shape
    #pool1 = sess.run(convo_1_pooling, feed_dict={input_x:batch_x})
    #print pool1.shape
    #relu1 = sess.run(convo_1_relu, feed_dict={input_x:batch_x})
    #print relu1.shape

    #plot_filters(0, pool1, 1, 'hot')
    #plot_filters(1, relu1, 1, 'hot')
    #raw_input()
