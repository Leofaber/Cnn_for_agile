import numpy as np
from astropy.io import fits
import sys
import os
from random import shuffle



title =     """
             @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    B I N A R Y   D A T A S E T   G E N E R A T O R
             @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            """



def get_label_from_filename(filename):
    if 'S' in filename:
        return 1
    else:
        return 0



def compute_max_value(image):
    max = image[0][0]
    for row in image:
        for elem in row:
            if elem > max:
                max = elem
    return max

def compute_min_value(image):
    min = image[0][0]
    for row in image:
        for elem in row:
            if elem < min:
                min = elem
    return min
#TODO
def linear_stretching(image, new_min = 0,new_max = 255):
    min = compute_min_value(image)
    max = compute_max_value(image)
    print 'min',min,'max',max,' => min',new_min,'max',new_max

    slope = (new_max - new_min) / ( max - min )
    print "slope", slope
    raw_input("..")
    stretched_image = np.zeros(image.shape, dtype=np.float64)

    """for i in range(stretched_image.shape[0]):
        for j in range(stretched_image.shape[1]):
            stretched_image[i,j] = slope * ( stretched_image[i,j] - min ) + new_min
            print image[i,j],"->",stretched_image[i,j]

    raw_input("..")"""


    return image


def create_binary_files(number_of_train_examples, number_of_test_examples, batch_size = 2000, mode = 'TRAIN', reduce_images_dimension_by = 0, dest_dataset_path = 'BATCH_DATASET',do_linear_stretching=False):


    sample_offset = 0
    number_of_examples = number_of_train_examples

    train_images_name = 'train_images_'
    train_labels_name = 'train_labels_'
    test_images_name = 'test_images_'
    test_labels_name = 'test_labels_'

    images_name = dest_dataset_path+'training_set/'+train_images_name
    labels_name = dest_dataset_path+'training_set/'+train_labels_name

    if mode == 'TEST' :
        sample_offset = number_of_train_examples # parte da l'ultimo training sample
        number_of_examples = number_of_test_examples
        images_name = dest_dataset_path+'test_set/'+test_images_name
        labels_name = dest_dataset_path+'test_set/'+test_labels_name


    images = np.zeros(tensors_shape, dtype=np.float64)
    labels = np.zeros(batch_size, dtype=np.int32)

    batch_limit = 0
    count = -1
    for i in range(number_of_examples):

        sample_name = samples[ sample_offset + i ]

        # Aggiungere le labels
        labels[batch_limit] = get_label_from_filename( sample_name )

        # Aggiungere le matrici
        hdulist = fits.open( dataset_path + sample_name )

        # Get the data  ->  TODO normalization
        image = np.zeros(hdulist[0].data.shape, dtype=np.float64)
        image = hdulist[0].data



        """ Reducing images size """
        if reduce_images_dimension_by > 0:

            img_rows = image.shape[0] # 100
            img_cols = image.shape[1] # 100

            new_img_rows = img_rows - reduce_images_dimension_by # 96
            new_img_cols = img_cols - reduce_images_dimension_by # 96

            reduce_images_dimension_by_for_size = reduce_images_dimension_by/2 # 2    x x 1 1 1 1 1 1 1 1 x x

            row_idx = np.arange( reduce_images_dimension_by_for_size, img_rows-reduce_images_dimension_by_for_size )
            col_idx = np.arange( reduce_images_dimension_by_for_size, img_cols-reduce_images_dimension_by_for_size )

            """https://stackoverflow.com/questions/22927181/selecting-specific-rows-and-columns-from-numpy-array"""
            images[batch_limit] = image[row_idx[:, None], col_idx]

        else:
            images[batch_limit] = image
        hdulist.close()

         # controllare se il batch e pieno
        if batch_limit == batch_size-1:
            batch_limit = 0
            count += 1
            #start = str(i-(batch_size-1))
            #stop = str(i)
            batch_images_name = images_name+str(count)+'.npy'
            batch_labels_name = labels_name+str(count)+'.npy'

            np.save(batch_images_name, images)
            np.save(batch_labels_name, labels)


            images = np.zeros(tensors_shape, dtype=np.float64)
            labels = np.zeros(batch_size, dtype=np.int32)
            print 'Batch',batch_images_name,'and',batch_labels_name,'completed and saved.'
        else:
            batch_limit += 1










"""
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                            M   A   I   N
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""

print title

#dataset_path = sys.argv[0]
#batch_size = sys.argv[1]
#reduce_images_dimension_by = sys.argv[2]

#dataset_path = 'dataset/'
#dest_dataset_path = 'batch_dataset'
dataset_path = 'dataset_smooth/'
dest_dataset_path = 'batch_smoothed_dataset_normalized/'
do_linear_stretching = True

batch_size = 10000
reduce_images_dimension_by = 4


# reading the filenames of each class
B_samples = [name for name in os.listdir(dataset_path) if 'B' in name]
S_samples = [name for name in os.listdir(dataset_path) if 'S' in name]
samples = []

B_size = len(B_samples)
S_size = len(S_samples)

# Putting all the samples in one array, first class1, second class2 etcetc..
tmp = 0
if B_size <= S_size:
    tmp = B_size
else:
    tmp = S_size
for i in range(tmp):
    samples.append(B_samples[i])
    samples.append(S_samples[i])
"""
    samples = [B,S,B,S, ....] 400.000
"""
#for i in range(10):
#    print samples[i]

dataset_size = len(samples)

# The test set size should be 1/8 of total
test_set_samples_number = dataset_size/8
training_set_samples_number = dataset_size-test_set_samples_number

print 'Dataset in:', dataset_path
print 'Dataset out:', dest_dataset_path
print 'Number of samples:', dataset_size
print 'Number of B samples:', B_size
print 'Number of S samples:', S_size
print 'Number of samples for training:', training_set_samples_number, '(',training_set_samples_number/2,'B, ',training_set_samples_number/2,'S )'
print 'Number of samples for testing:', test_set_samples_number, '(',test_set_samples_number/2,'B, ',test_set_samples_number/2,'S )'




# Compute rows and cols of images and Tensors Numpy Shape
hdulist = fits.open(dataset_path+samples[0])
labels_vector_shape = 1, batch_size

"""
    The input layer (that contains the image) should be divisible by 2 many times.
    Common numbers include 32 (e.g. CIFAR-10), 64, 96 (e.g. STL-10), or 224
    (e.g. common ImageNet ConvNets), 384, and 512.
"""
rows = hdulist[0].data.shape[0]
cols = hdulist[0].data.shape[1]
new_rows = rows - reduce_images_dimension_by
new_cols = rows - reduce_images_dimension_by
tensors_shape = batch_size, new_rows, new_cols # (2000, 96, 96)

if reduce_images_dimension_by > 0:
    print "Images will be resized from:",rows,"x",cols,"to",new_rows,"x",new_cols

print "Tensors shape:", tensors_shape


print training_set_samples_number / batch_size," train batch files will be created"
print test_set_samples_number / batch_size," test batch files will be created"

raw_input("Press any key to start..")

""" Cereating folders """
if not os.path.exists(dest_dataset_path):
    os.makedirs(dest_dataset_path)

if not os.path.exists(dest_dataset_path+"/training_set"):
    os.makedirs(dest_dataset_path+"/training_set")

if not os.path.exists(dest_dataset_path+"/test_set"):
    os.makedirs(dest_dataset_path+"/test_set")

""" Saving images in batches and writing them """
create_binary_files( training_set_samples_number, test_set_samples_number, batch_size, 'TRAIN', reduce_images_dimension_by, dest_dataset_path,linear_stretching)
create_binary_files( test_set_samples_number, test_set_samples_number, batch_size, 'TEST', reduce_images_dimension_by, dest_dataset_path,linear_stretching)
