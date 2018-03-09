# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 19:30:19 2016

@author: jmiller
"""

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import numpy as np
from glob import glob
import cv2
import os
import sys
import datetime as dt
import imp
from functools import reduce
util = imp.load_source(
    'dl_util',
    '/home/jmiller/Dropbox/cnn_stuff/code/dl_util.py')
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from img_iterator import *


def lst_shuffle(lst, num=None, tags=None):
    if num is None:
        num = len(lst)
    index_arr = np.random.randint(0, len(lst), num)
    shuffled_lst = [None] * num
    # np.random.shuffle(index_arr)
    for i in xrange(num):
        shuffled_lst[i] = lst[index_arr[i]]
    if tags is not None:
        # print(len(tags))
        shuffled_tags = [None] * num
        for i in xrange(num):
            shuffled_tags[i] = tags[index_arr[i]]
    if tags is None:
        return shuffled_lst
    else:
        return shuffled_lst, shuffled_tags


def arr_shuffle(imgs, tags):
    index_arr = np.arange(tags.shape[0])
    # print(index_arr.shape)
    shuffled_imgs = np.empty(imgs.shape)
    shuffled_tags = np.empty(tags.shape)
    np.random.shuffle(index_arr)
    for i in xrange(imgs.shape[0]):
        shuffled_tags[i] = tags[index_arr[i]]
        shuffled_imgs[i, :] = imgs[index_arr[i], :]

    return shuffled_imgs, shuffled_tags


def writeTxt(path, lst1, lst2, lst3):
    with open(path, 'w') as txt:
        for i in xrange(len(lst1)):
            txt.write('Tag: ' + str(lst2[i]) + ' : ')
            txt.write('Pred: ' + str(lst3[i]) + ' : ')
            txt.write(lst1[i])
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


def imgStream(img_lst, batch_size=20):
    '''
    Takes a list of paths to images and a batch size
    Returns a batch size number of images
    '''
    i = 0
    j = batch_size

    # Check to make sue the batch_size works with the number of images
    if (len(img_lst) / float(batch_size)).is_integer() == False:
        raise NameError(
            'The batch size does not divide into the number of images evenly. Try changing the batch size.')
    while j < (len(img_lst) + 1):
        tmp = img_lst[i:j]
        i = j
        j += batch_size
        yield tmp


def class_predictions(model, img_lst, batch_size=20):
    '''
    Takes a compiled keras model and list of image paths.
    Returns the predicted class for the images.
    '''
    # Initialize img stream
    path_lst = imgStream(img_lst, batch_size=batch_size)
    preds = []
    for lst in path_lst:
        imgs = util.load_images(lst)
        for pred in model.predict_classes(
                imgs, batch_size=batch_size, verbose=1):
            preds.append(pred)

    return preds


def plot_confusion_matrix(
        con_matrix,
        prefix,
        title='Confusion matrix',
        cmap=plt.cm.Blues):
    fig, ax = plt.subplots()
    ax.imshow(con_matrix, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    # fig.colorbar(con_matrix)
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Bands', 'Not Bands'], rotation=45)
    plt.yticks(tick_marks, ['Bands', 'Not Bands'])
    plt.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    # ax.annotate('{:.2f}%'.format(con_matrix[0,0]),xy=(-0.4,0.4),xytext=(-0.4,0.4))
    # ax.annotate('{:.2f}%'.format(con_matrix[0,1]),xy=(0.6,0.4),xytext=(0.6,0.4))
    # ax.annotate('{:.2f}%'.format(con_matrix[1,0]),xy=(-0.4,1.4),xytext=(-0.4,1.4))
    # ax.annotate('{:.2f}%'.format(con_matrix[1,1]),xy=(0.6,1.4),xytext=(0.6,1.4))
    plt.savefig('{}_confusion_matrix2.png'.format(prefix))


def factor(n, max_=50):
    '''
    Takes number and returns the highest factor that is less than max_.
    '''
    factors = sorted(reduce(
        list.__add__, ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
    tmp = 1
    for item in factors:
        if item > tmp and item < max_:
            tmp = item
    return tmp


if __name__ == '__main__':

    home = os.environ['HOME']
    json_path = '/home/jmiller/Dropbox/cnn_stuff/models/040117/040117-1151/model_ftil_25_040117-1151.json'
    # weights_path=home+'/Dropbox/cnn_stuff/models/100616/transverseVGG_ftil_13_weights_100616-1136.h5'
    weights_path = '/home/jmiller/Dropbox/cnn_stuff/models/040117/040117-1151/transVGG.hdf5'
    test_data_dir = home + '/Dropbox/soni_images/data/test'
    # test_data_dir='/media/jmiller/ubuntu_storage/training_stuff/test'
    weights_name = weights_path.split('/')[-1]
    print('###### Using {0} ######'.format(weights_name))

    model = model_from_json(open(json_path, 'r').read())
    model.load_weights(weights_path)

    model.compile(optimizer='SGD',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Get number of images in directory to determine optimum batch size
    nb_bands = len(glob(test_data_dir + '/bands/*'))
    nb_nbands = len(glob(test_data_dir + '/not_bands/*'))
    batch_size = factor(nb_bands + nb_nbands)

    datagen = ImageDataGenerator(
        rescale=1. / 255)
    test_gen = datagen.flow_from_directory(
        test_data_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='binary')

    batch_size = factor(test_gen.N)
    current_index = 0
    proceed = True
    predictions = []
    tags = []
    num_batches = test_gen.N / test_gen.batch_size
    print num_batches
    for i in xrange(num_batches):
        current_index = (test_gen.batch_index *
                         test_gen.batch_size) % test_gen.N
        if test_gen.N < (current_index + batch_size):
            break
        images = test_gen.next()
        tags.append(images[1])
        predictions.append(model.predict_classes(images[0]))

    preds = []
    labels = []
    for lst in predictions:
        for item in lst:
            preds.append(item)
    for lst in tags:
        for item in lst:
            labels.append(item)

    # Generate confusion matrix and plot
    pred_tags = preds
    cm = confusion_matrix(labels, pred_tags)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Confusion matrix: {0}'.format(cm))
    print('Normalized confusion matrix: {0}'.format(cm_normalized))
    plot_confusion_matrix(cm_normalized, 'Normalized')
    plot_confusion_matrix(cm, 'Standard')

    tmp_labels = np_utils.to_categorical(labels, 2)
    tmp_preds = np_utils.to_categorical(preds, 2)
    print(classification_report(tmp_labels, tmp_preds))

    #precision = dict()
    #recall = dict()
    #average_precision = dict()

    precision, recall, _ = precision_recall_curve(tmp_labels, tmp_preds)
    # average_precision=average_precision_score(labels,preds)
    '''
  plt.clf()
  plt.plot(recall, precision, lw=2, color='navy',
           label='Precision-Recall curve')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.ylim([0.0, 1.05])z
  plt.xlim([0.0, 1.0])
  plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
  plt.legend(loc="lower left")
  plt.show()









  #write misclassified images to text file
  with open('incorrect_classifications.txt','w') as txt:
      txt.write('Bands incorrectly classified: \n')
      for line in band_bad:
          txt.write(line)
          txt.write('\n')
      txt.write('Not bands incorrectly classified: \n')
      for line in not_band_bad:
          txt.write(line)
          txt.write('\n')






  test_datagen = IDG(rescale=1./255)

  test_generator = test_datagen.flow_from_directory(
                  test_data_dir,
                  target_size=(256,256),
                  batch_size=batch_size,
                  class_mode='binary')


  evaluation=model.evaluate_generator(
                  test_generator,
                  val_samples=500)

  print evaluation
  '''
