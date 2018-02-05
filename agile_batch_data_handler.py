import os
import numpy as np

class Dataset:
    def __init__(self, name, training_set_path, test_set_path):

        print name
        self.name = name

        self.training_set_path = training_set_path
        self.test_set_path = test_set_path

        self.training_set_size = len( [name for name in os.listdir(self.training_set_path) if 'images' in name ])
        self.test_set_size = len( [name for name in os.listdir(self.test_set_path) if 'images' in name ])



        self.train_batch_images = []                # samples [ ... ]
        self.train_batch_labels = []                # labels [ ... ]
        self.current_train_batch_index = -1         # the current batch file
        self.train_total_images_per_binary_file = 0 # the total number of samples in the binary file
        self.train_batch_consumed = 0               # count how many examples I used from the loaded binary file

        self.test_set_images= []
        self.test_set_labels = []
        self.test_set_loaded = False
        self.test_batch_images = []
        self.test_batch_labels = []
        self.current_test_batch_index = -1
        self.test_total_images_per_binary_file = 0
        #self.test_batch_consumed = 0

        print "Loading train and test set.."
        if not self.load_new_batch('TRAIN') or not self.load_new_batch('TEST'):
            print "Can't load train/test binary file"
        else:
            self.train_total_images_per_binary_file = self.train_batch_images.shape[0]
            self.test_total_images_per_binary_file = self.test_batch_images.shape[0]

        if not self.load_test_set():
            print "Can't load test set"



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

    def count_classes_test_set_labels(self,labels):
        countS = 0
        countB = 0
        for label in labels:
            if label[0] == 1:
                countS += 1
            elif label[0] == 0:
                countB += 1
        return countS, countB

    """
        Example: 3 different classes (0,1,2) and 5 samples:
        [ 2, 1, 0, 2, 1 ] => [ [0,0,1], [0,1,0], [1,0,0], [0,0,1], [0,1,0] ]
    """
    def convert_to_class_array(self, labels_array):
        number_of_classes = int(max(labels_array)+1) # because the min is 0
        number_of_sample = int(len(labels_array))

        new_labels_array = np.zeros([number_of_sample, number_of_classes])
        index = 0

        for label in labels_array:
            a = np.zeros(number_of_classes)
            a[int(label)] = 1
            new_labels_array[index] = a
            index += 1

        return new_labels_array

    def get_number_of_training_samples(self):
        return self.training_set_size*self.train_total_images_per_binary_file

    def get_number_of_test_samples(self):
        return self.test_set_size*self.test_total_images_per_binary_file

    def check_load(self):
        try:
            #print "\n@@@@@@@@@@@@@@@@@@@@@@@@@@"
            #print "@@@", self.name, "info","@@@"
            #print "@@@@@@@@@@@@@@@@@@@@@@@@@@"
            print "\nDataset check():",self.name
            print " Training set batches number:",self.training_set_size
            print " Test set batches number:",self.test_set_size

            print "\nShape of training images:", self.train_batch_images.shape
            print "Number of pixels of training image 0:", self.sum_img_pixels(self.train_batch_images[0])
            print "Number of pixels of training image 1:", self.sum_img_pixels(self.train_batch_images[1])

            print " Training labels vector:",self.train_batch_labels
            print " Shape of training labels vector:", self.train_batch_labels.shape
            counts = self.count_classes(self.train_batch_labels)
            print " Labels vector:\n   #True:",counts[0],"\n   #False:",counts[1]


            print "\nShape of test images:", self.test_set_images.shape
            print "Number of pixels of test image 0:", self.sum_img_pixels(self.test_set_images[0])
            print "Number of pixels of test image 1:", self.sum_img_pixels(self.test_set_images[1])
            print "Number of pixels of test image 2:", self.sum_img_pixels(self.test_set_images[2])
            print "Number of pixels of test image 3:", self.sum_img_pixels(self.test_set_images[3])
            print "Number of pixels of test image 4:", self.sum_img_pixels(self.test_set_images[4])
            print "Number of pixels of test image 5:", self.sum_img_pixels(self.test_set_images[5])
            print " Labels vector:",self.test_set_labels[0],self.test_set_labels[1],self.test_set_labels[2],self.test_set_labels[3],self.test_set_labels[4],self.test_set_labels[5]
            print " Shape of test labels vector:", self.test_set_labels.shape
            counts = self.count_classes_test_set_labels(self.test_set_labels)
            print " Labels vector:\n   #True:",counts[0],"\n   #False:",counts[1]
            return True
        except Exception as e:
            print "Error: ",e
            return False

    """ Return a couple (images, labels) that is a portion of the self.train_batch """
    def get_batch(self, batch_size, mode = 'TRAIN'):

        batch_part = ()

        if mode == 'TRAIN':

            """ Handling errors """

            if batch_size > self.train_total_images_per_binary_file:
                print "Error! You are requesting too much train data! Max request = ", self.train_total_images_per_binary_file
                return ()

            if self.train_total_images_per_binary_file % batch_size != 0:
                print "Error! Please enter a batch_size that's a divisor of ", self.train_total_images_per_binary_file
                return ()

            if self.train_batch_consumed >= self.train_total_images_per_binary_file:
                if not self.load_new_batch('TRAIN'):
                    print "Sorry! No more data is available. You requested",self.current_train_batch_index,"files"
                    return ()
                self.train_batch_consumed = 0

            # Take a piece of the loaded samples file
            samples_batch = self.train_batch_images[self.train_batch_consumed:self.train_batch_consumed + batch_size]
            # Take a piece of the loaded labels file
            labels_batch = self.train_batch_labels[self.train_batch_consumed:self.train_batch_consumed + batch_size]

            batch_part = samples_batch, self.convert_to_class_array(labels_batch)

            # Increment the 'consumed' attribute
            self.train_batch_consumed += batch_size
            #print "\nbatch consumed: ", self.train_batch_consumed,"/",self.train_total_images_per_binary_file
            return batch_part[0], batch_part[1]

        # gives all test samples
        elif mode == 'TEST':

            print "Deprecated. Please call get_test_set() instead."








    def load_new_batch(self, mode = 'TRAIN'):


        if mode == 'TRAIN':

            # Se non ci sono piu' batches
            if self.current_train_batch_index == self.training_set_size-1:
                print "No more batches are available"
                return False

            self.current_train_batch_index += 1
            #print "Loading new train batch. Part =",self.current_train_batch_index

            self.train_batch_images = \
                np.load( self.training_set_path + 'train_images_' + str(self.current_train_batch_index) + '.npy')

            # ( ? , rows, cols) -> Grey scale images (?, rows, cols, 1)
            self.train_batch_images = np.expand_dims(self.train_batch_images, axis=3)

            self.train_batch_labels = \
                np.load( self.training_set_path + 'train_labels_' + str(self.current_train_batch_index) + '.npy')

            self.train_batch_consumed = 0

        elif mode == 'TEST':

            # Se non ci sono piu' batches
            if self.current_test_batch_index == self.test_set_size-1:
                print "No more test batches are available"
                return False

            self.current_test_batch_index += 1
            #print "Loading new test batch. Part =",self.current_test_batch_index

            self.test_batch_images = \
                np.load( self.test_set_path + 'test_images_' + str(self.current_test_batch_index) + '.npy')

            # ( ? , rows, cols) -> Grey scale images (?, rows, cols, 1)
            self.test_batch_images = np.expand_dims(self.test_batch_images, axis=3)

            self.test_batch_labels = \
                np.load( self.test_set_path + 'test_labels_' + str(self.current_test_batch_index) + '.npy')

            self.test_batch_consumed = 0


        return True


    def get_test_set(self):
        if not self.test_set_loaded:
            self.load_test_set()
        return self.test_set_images,self.test_set_labels

    def get_number_of_test_samples(self):
        return self.test_set_images.shape[0]

    def load_test_set(self):

        limit = 4

        total = self.test_set_size * self.test_total_images_per_binary_file

        #self.test_set_images = np.zeros( (50000,96,96,1), dtype = np.float64)
        #self.test_set_labels = np.zeros( (50000, 2), dtype = np.float64)

        #print test_set_images.shape
        #print test_set_labels.shape


        to_concat_images = []
        to_concat_labels = []
        to_concat_images.append(self.test_batch_images)
        to_concat_labels.append(self.test_batch_labels)

        for i in range(self.test_set_size-1-limit):
            self.load_new_batch('TEST')
            to_concat_images.append(self.test_batch_images)
            to_concat_labels.append(self.test_batch_labels)

        self.test_set_images = to_concat_images[0]
        self.test_set_labels = self.convert_to_class_array(to_concat_labels[0])

        i = 1
        while i < self.test_set_size-limit:
            self.test_set_images = np.concatenate((self.test_set_images,to_concat_images[i]), axis=0)
            self.test_set_labels = np.concatenate((self.test_set_labels,self.convert_to_class_array(to_concat_labels[i])), axis=0)
            i += 1



        self.test_set_loaded = True

        return True
