import os
import numpy as np

class Dataset:
    def __init__(self, name, batch_size, training_set_path, test_set_path):

        self.name = name

        self.training_set_path = training_set_path
        self.test_set_path = test_set_path

        self.training_set_size = len( [name for name in os.listdir(self.training_set_path) ])
        self.test_set_size = len( [name for name in os.listdir(self.test_set_path) ])


        self.batch_size = batch_size

        self.train_batch_images = []                # samples [ ... ]
        self.train_batch_labels = []                # labels [ ... ]
        self.current_train_batch_index = -1         # the current batch file
        self.train_batch_consumed = 0               # count how many examples I used from the loaded binary file
        self.train_total_images_per_binary_file = 0 # the total number of samples in the binary file

        self.test_batch_images= []
        self.test_batch_labels = []
        self.current_test_batch_index = -1
        self.test_batch_consumed = 0
        self.test_total_images_per_binary_file = 0

        if not self.load_new_batch('TRAIN') or not self.load_new_batch('TEST'):
            print "Can't load train/test binary file"
        else:
            self.train_total_images_per_binary_file = self.train_batch_images.shape[0]
            self.test_total_images_per_binary_file = self.test_batch_images.shape[0]


    # For debug purpose
    def sum_img_pixels(self,img):
        sum_pixels = 0
        for row in img:
            for element in row:
                sum_pixels += element
        return sum_pixels

    # For debug purpose
    def count_classes(self,labels):
        countS = 0
        countB = 0
        for label in labels:
            if label == 1:
                countS += 1
            elif label == 0:
                countB += 1
        return countS, countB


    def check_load(self):
        print "\n@@@@@@@@@@@@@@@@@@@@@@@@@@"
        print "@@@", self.name, "info","@@@"
        print "@@@@@@@@@@@@@@@@@@@@@@@@@@"

        print "Training set batches number:",self.training_set_size
        print "Test set batches number:",self.test_set_size
        print "Batch size:",self.batch_size

        print "\nShape of training images:", self.train_batch_images.shape
        #print "Number of pixels of training image 0:", self.sum_img_pixels(self.train_batch_images[0])
        #print "Number of pixels of training image 1:", self.sum_img_pixels(self.train_batch_images[1])

        print "Training labels vector:",self.train_batch_labels[0]
        print "Shape of training labels vector:", self.train_batch_labels[0].shape
        counts = self.count_classes(self.train_batch_labels[0])
        print "Labels vector:\n#True:",counts[0],"\n#False:",counts[1]


        print "\nShape of test images:", self.test_batch_images.shape
        #print "Number of pixels of test image 0:", self.sum_img_pixels(self.test_batch_images[0])
        #print "Number of pixels of test image 1:", self.sum_img_pixels(self.test_batch_images[1])

        print "Labels vector:",self.test_batch_labels[0]
        print "Shape of test labels vector:", self.test_batch_labels[0].shape
        counts = self.count_classes(self.test_batch_labels[0])
        print "Labels vector:\n#True:",counts[0],"\n#False:",counts[1]


    """ Return a couple (images, labels) that is a portion of the self.train_batch """
    def get_batch(self, mode = 'TRAIN'):

        batch_part = ()

        if mode == 'TRAIN':

            if self.train_batch_consumed >= self.train_total_images_per_binary_file:
                if not self.load_new_batch('TRAIN'):
                    return ()
                self.train_batch_consumed = 0


            batch_part = \
                    self.train_batch_images[self.train_batch_consumed:self.train_batch_consumed + self.batch_size], \
                    self.train_batch_labels[self.train_batch_consumed:self.train_batch_consumed + self.batch_size]

            self.train_batch_consumed += self.batch_size
            print mode,"\nbatch consumed: ", self.train_batch_consumed,"/",self.train_total_images_per_binary_file


        elif mode == 'TEST':

            if self.test_batch_consumed >= self.test_total_images_per_binary_file:
                if not self.load_new_batch('TEST'):
                    return ()
                self.test_batch_consumed = 0


            batch_part = \
                    self.test_batch_images[self.test_batch_consumed: self.test_batch_consumed + self.batch_size], \
                    self.test_batch_labels[self.test_batch_consumed: self.test_batch_consumed + self.batch_size]

            self.test_batch_consumed += self.batch_size
            print mode,"\nbatch consumed: ", self.test_batch_consumed,"/",self.test_total_images_per_binary_file


        return batch_part



    def load_new_batch(self, mode = 'TRAIN'):

        # Se non ci sono piu' batches
        if self.current_train_batch_index == self.training_set_size-1:
            print "Error: no more batches are available"
            return False


        if mode == 'TRAIN':
            print "Loading new train batch"
            self.current_train_batch_index += 1

            self.train_batch_images = \
                np.load( self.training_set_path + 'train_images_' + str(self.current_train_batch_index) + '.npy')

            self.train_batch_labels = \
                np.load( self.training_set_path + 'train_labels_' + str(self.current_train_batch_index) + '.npy'),

            self.train_batch_consumed = 0

        elif mode == 'TEST':
            print "Loading new test batch"
            self.current_test_batch_index += 1

            self.test_batch_images = \
                np.load( self.test_set_path + 'test_images_' + str(self.current_test_batch_index) + '.npy')

            self.test_batch_labels = \
                np.load( self.test_set_path + 'test_labels_' + str(self.current_test_batch_index) + '.npy'),

            self.test_batch_consumed = 0


        return True
