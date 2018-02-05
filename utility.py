import sys
from math import ceil
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg') # TkAgg works but it's slow
                  # GTK doesnt work

"""
[u'pgf', u'cairo', u'MacOSX', u'CocoaAgg', u'gdk', u'ps', u'GTKAgg', u'nbAgg', u'GTK',
u'Qt5Agg', u'template', u'emf', u'GTK3Cairo', u'GTK3Agg', u'WX', u'Qt4Agg', u'TkAgg',
u'agg', u'svg', u'GTKCairo', u'WXAgg', u'WebAgg', u'pdf'
"""
import matplotlib.pyplot as plt

def print_2D_matrix_on_console(matrix):
    for row in matrix:
        for elem in row:
            sys.stdout.write(str(elem))
        print ""

def plot_image_with_filters(window_no, image, image_number, conv2D_output, cmap_modes=['grey','grey']):
    start_index = len(range(window_no+1))
    plot_image(start_index, image, cmap_modes[0])
    plot_filters(start_index+1, conv2D_output, image_number, cmap_modes[1])
    raw_input()

def plot_image(figure_no, image, cmap_mode):
    img = plt.figure(figure_no, figsize=(50,50))
    plt.imshow(image[:,:,0], cmap=cmap_mode)
    img.show()

def plot_filters(figure_no, conv2D_output, image_number, cmap_mode):
    filters = conv2D_output.shape[3]
    f = plt.figure(figure_no, figsize=(20,20))
    n_columns = 6
    n_rows = ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.imshow(conv2D_output[image_number,:,:,i], cmap= cmap_mode, interpolation="nearest")
    f.show()

"""cmap possibile values:

Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap,
CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd,
OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r,
Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r,
PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu,
RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r,
Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr,
YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r,
bone, bone_r, brg, brg_r, bwr, bwr_r, cool, cool_r, coolwarm, coolwarm_r, copper,
copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray,
gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, seismic, seismic_r, spectral, spectral_r, spring, spring_r, summer, summer_r, terrain, terrain_r, viridis, viridis_r, winter, winter_r
"""
