# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 18:15:25 2016

@author: jmiller
"""

import numpy as np
import cPickle as pkl
from glob import glob
import cv2
import shutil
import os
import sys
import datetime as dt
import imp
#util = imp.load_source('dl_util','/Users/jmiller/Dropbox/cnn_stuff/dl_util.py')


def lst_shuffle(lst, num=None, tags=None):
    if num is None:
        num = len(lst)
    index_arr = np.random.randint(0, len(lst), num)
    shuffled_lst = [None] * num
    # np.random.shuffle(index_arr)
    for i in xrange(num):
        shuffled_lst[i] = lst[index_arr[i]]
    if tags is not None:
        print(len(tags))
        shuffled_tags = [None] * num
        for i in xrange(num):
            shuffled_tags[i] = tags[index_arr[i]]
    if tags is None:
        return shuffled_lst
    else:
        return shuffled_lst, shuffled_tags


def arr_shuffle(imgs, tags):
    index_arr = np.arange(tags.shape[0])
    print(index_arr.shape)
    shuffled_imgs = np.empty(imgs.shape)
    shuffled_tags = np.empty(tags.shape)
    np.random.shuffle(index_arr)
    for i in xrange(imgs.shape[0]):
        shuffled_tags[i] = tags[index_arr[i]]
        shuffled_imgs[i, :] = imgs[index_arr[i], :]

    return shuffled_imgs, shuffled_tags


def writeTxt(path, lst1=None, lst2=None, stat=None):
    if stat is not None:
        with open(path, 'a') as txt:
            txt.write('Total number of bands found: {0}!'.format(str(stat)))
    else:
        with open(path, 'a') as txt:
            for img_path in lst1:
               # txt.write('Tag: '+str(lst2[i])+' : ')
               # txt.write('Pred: '+str(lst2[i])+' : ')
                txt.write(img_path)
                txt.write('\n')


def load_images(file_list, imgs=None, num_images=None, shuffle=False):
    '''
    Loads the images and whitens them with ZCA whitening.
    Returns a mxn matrix where m is the number of images and n is the number of
        pixels per image.
    '''
    if num_images is None:
        num_images = len(file_list)

    if shuffle:
        file_list = lst_shuffle(file_list, num_images)

    for i in xrange(len(file_list)):
        # print(i)
        # print(file_list)
        img = cv2.imread(file_list[i])
        try:
            img.shape
        except AttributeError:
            print(file_list[i])
        if img.shape[0] != 256:
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 1.
        img = img.astype('float32')
        img = np.rollaxis(img, 2)
        img = np.expand_dims(img, axis=0)
        if imgs is not None:
            imgs = np.vstack([imgs, img])
        else:
            if i == 0:
                images_matrix = img
            if i != 0:
                images_matrix = np.vstack([images_matrix, img])

    if imgs is not None:
        images_matrix = imgs
        return images_matrix, file_list
    else:
        return images_matrix
