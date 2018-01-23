import numpy as np
from astropy.io import fits
import sys
import os
from random import shuffle



"""
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            D A T A S E T   B A T C H   G E N E R A T O R
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""






def get_label_from_filename(filename):
    if 'B' in filename:
        return True
    else:
        return False


def create_batch_files(number_of_train_examples, number_of_test_examples, batch_size = 1000, mode = 'TRAIN'):


    sample_offset = 0
    number_of_examples = number_of_train_examples
    images_name = 'training_set/train_images_'
    labels_name = 'training_set/train_labels_'
    if mode == 'TEST' :
        sample_offset = number_of_train_examples
        number_of_examples = number_of_test_examples
        images_name = 'test_set/test_images_'
        labels_name = 'test_set/test_labels_'


    labels = np.zeros(batch_size, dtype=bool)
    images = np.zeros(tensors_shape, dtype=int)

    batch_limit = 0
    for i in range(number_of_examples):

        sample_name = samples[ sample_offset + i ]

        # Aggiungere le labels
        np.append(labels, get_label_from_filename( sample_name ) )

        # Aggiungere le matrici
        hdulist = fits.open( dataset_path + sample_name )
        images[batch_limit] = hdulist[0].data
        hdulist.close()

         # controllare se il batch e pieno
        if batch_limit == batch_size-1:
            batch_limit = 0
            start = str(i-(batch_size-1))
            stop = str(i)
            batch_images_name = images_name+start+'-'+stop+'.npy'
            batch_labels_name = labels_name+start+'-'+stop+'.npy'
            np.save(batch_images_name, images)
            np.save(batch_labels_name, labels)
            labels = np.zeros(batch_size, dtype=bool)
            images = np.zeros(tensors_shape, dtype=int)
            print 'Batch',batch_images_name,'and',batch_labels_name,'completed and saved.'
        else:
            batch_limit += 1










"""
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                            M   A   I   N
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""


#dataset_path = sys.argv[0]
#batch_size = sys.argv[1]

dataset_path = 'dataset/'
batch_size = 1000

B_samples = [name for name in os.listdir(dataset_path) if 'B' in name]
S_samples = [name for name in os.listdir(dataset_path) if 'S' in name]
samples = []

B_size = len(B_samples)
S_size = len(S_samples)

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

dataset_size = len(samples)


test_set_samples_number = dataset_size/4
training_set_samples_number = dataset_size-test_set_samples_number

print 'Number of samples:', dataset_size
print 'Number of B samples:', B_size
print 'Number of S samples:', S_size
print 'Number of samples for training:', training_set_samples_number, '(',training_set_samples_number/2,'B, ',training_set_samples_number/2,'S )'
print 'Number of samples for testing:', test_set_samples_number, '(',test_set_samples_number/2,'B, ',test_set_samples_number/2,'S )'

# Compute rows and cols of images and Tensors Numpy Shape
hdulist = fits.open(dataset_path+samples[0])
tensors_shape = batch_size, hdulist[0].data.shape[0], hdulist[0].data.shape[1]
print "Tensors shape:", tensors_shape


if not os.path.exists("training_set"):
    os.makedirs("training_set")

if not os.path.exists("test_set"):
    os.makedirs("test_set")

raw_input("Press any key to start..")


create_batch_files(training_set_samples_number/2,test_set_samples_number/2,batch_size, 'TRAIN')
create_batch_files(test_set_samples_number/2,test_set_samples_number/2,batch_size, 'TEST')
