# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 18:17:50 2016

@author: J.J. Miller
"""
import matplotlib.pyplot as plt
import numpy as numpy
from glob import glob
import os
import string


def plot_multiImage(rowsCols, path1=None, path2=None):
    '''
    rowsCols: Tuple with number of (rows,columns)
    path1: Images for first row
    path2: Images for second row
    '''
    home = os.environ['HOME']
    path1 = home + '/Dropbox/cnn_stuff/images/keras_val/bands/hold/*'
    path2 = home + '/Dropbox/cnn_stuff/images/keras_val/not_bands/hold/*'
    one = glob(path1)
    two = glob(path2)
    for k in two:
        one.append(k)
    alphabet = list(string.ascii_lowercase)

    fig, axarr = plt.subplots(nrows=rowsCols[0], ncols=rowsCols[1])
    t = 0
    for i in xrange(rowsCols[0]):
        for j in xrange(rowsCols[1]):
            axarr[i, j].imshow(plt.imread(one[t]))
            axarr[i, j].tick_params(
                axis='x',
                which='both',
                bottom='off',
                top='off',
                labelbottom='off')
            axarr[i, j].axes.get_yaxis().set_visible(False)
            axarr[i, j].set_xlabel(alphabet[t])
            t += 1
    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_xticklabels() for a in axarr[1, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[1, :]], visible=False)

    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    plot_multiImage((2, 3))

'''
axarr[0,1].imshow(plt.imread(bands_path[1]))
axarr[0,1].tick_params(
	axis='x',
	which='both',
	bottom='off',
	top='off',
	labelbottom='off')
axarr[0,1].axes.get_yaxis().set_visible(False)
axarr[0,1].set_xlabel('b')


axarr[0,2].imshow(plt.imread(bands_path[2]))
axarr[0,2].tick_params(
	axis='x',
	which='both',
	bottom='off',
	top='off',
	labelbottom='off')
axarr[0,2].axes.get_yaxis().set_visible(False)
axarr[0,2].set_xlabel('c')


axarr[1,0].imshow(plt.imread(not_path[0]))
axarr[1,0].tick_params(
	axis='x',
	which='both',
	bottom='off',
	top='off',
	labelbottom='off')
axarr[1,0].axes.get_yaxis().set_visible(False)
axarr[1,0].set_xlabel('d')


axarr[1,1].imshow(plt.imread(not_path[1]))
axarr[1,1].tick_params(
	axis='x',
	which='both',
	bottom='off',
	top='off',
	labelbottom='off')
axarr[1,1].axes.get_yaxis().set_visible(False)
axarr[1,1].set_xlabel('e')


axarr[1,2].imshow(plt.imread(not_path[2]))
axarr[1,2].tick_params(
	axis='x',
	which='both',
	bottom='off',
	top='off',
	labelbottom='off')
axarr[1,2].axes.get_yaxis().set_visible(False)
axarr[1,2].set_xlabel('f')
'''
