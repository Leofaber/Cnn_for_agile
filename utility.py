import sys
import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt

def print_2D_matrix_on_console(matrix):
    for row in matrix:
        for elem in row:
            sys.stdout.write(str(elem))
        print ""

def print_2D_matrix_on_window(matrix, linear_stretch = False):

    stretched_matrix = np.zeros(matrix.shape, dtype=int)
    if linear_stretch:
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                if matrix[row][col] == 1:
                    stretched_matrix[row][col] = 255
                else:
                    stretched_matrix[row][col] = matrix[row][col]
        plt.matshow(stretched_matrix, cmap='hot')
    else:
        plt.matshow(matrix, cmap='hot')
    plt.show()
