# -*- coding: utf-8 -*-
# @Author: jmiller
# @Date:   2017-05-04 09:25:39
# @Last Modified by:   jmiller
# @Last Modified time: 2017-05-04 11:15:10

import numpy as np
#import re
import os
from PIL import Image
from glob import glob
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from img_iterator import Img_Iterator
from img_iterator import factor
import progressbar
import json
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool


def img_to_array(img):
    # image has dim_ordering (height, width, channel)
    x = np.asarray(img, dtype='float32')
    x = x.transpose(2, 0, 1)
    return x


def load_img(path, target_size=(256, 256)):
    '''Load an image into PIL format.
    # Arguments
      path: path to image file
      grayscale: boolean
      target_size: None (default to original size)
          or (img_height, img_width)
    '''
    img = Image.open(path)
    img = img.resize((target_size[1], target_size[0]))
    return img


def classify(model,
             directory,
             text='',
             verbose=0,
             copy=False):
    '''
    Takes a compiled model and directory of images and either returns a text
      file of the bands images or copys them to a new folder.
    destination: Directory where bands images will be copied if copy is True.
    text: path where text file will be saved for classified images.
    copy: Boolean. If True, it will copy bands images to the destination directory.
    '''

    datagen = ImageDataGenerator(rescale=1 / 255.)
    gen = datagen.flow_from_directory(
        directory=directory,
        target_size=(256, 256),
        batch_size=32,
        shuffle=False)

    ind = 0
    done = False
    b_index = 0
    band_files = []
    images = gen.filenames
    month = gen.directory.split('/')[-2]
    day = gen.directory.split('/')[-1]
    batchSize = factor(len(images))

    bar = progressbar.ProgressBar(
        redirect_stdout=True,
        max_value=(
            gen.N / gen.batch_size),
        widgets=[
            progressbar.Percentage(),
            ' ',
            '(',
            progressbar.SimpleProgress(),
            ')',
            ' Batch Size: {0}'.format(
                gen.batch_size),
            progressbar.Bar(),
            'Month: {0} '.format(month),
            'Day: {0} '.format(day),
            progressbar.ETA(),
        ])
    i = 0
    for batch in gen:
        predictions = model.predict_classes(batch[0],
                                            batch_size=gen.batch_size,
                                            verbose=0)
        for pred in predictions:
            if pred[0] == 0:
                band_files.append(images[ind])
            ind += 1
            if ind >= gen.N:
                done = True
                break
        if done:
            break
        bar.update(i)
        i += 1
    print('Images found: {0}'.format(len(band_files)))
    return band_files

    '''
    month = images.split('/')[-2]
    day = images.split('/')[-1]
    batchSize = factor(len(images))

    gen = Img_Iterator()
    stream = gen.flow_from_directory(images, batch_size=batchSize)
    i = 0
    j = []
    # Number of times the classifier will be run
    stop = int(stream.N / stream.batch_size)
    predictions = []
    band_files = []
    bar = progressbar.ProgressBar(redirect_stdout=True, max_value=stop,
    		widgets=[
    			progressbar.Percentage(), ' ',
    			'(', progressbar.SimpleProgress(), ')',
    			' Batch Size: {0}'.format(stream.batch_size),
    			progressbar.Bar(),
    			'Month: {0} '.format(month),
    			'Day: {0} '.format(day),
    			progressbar.ETA(),
			])
    for batch in stream:
        # Stop if number of batches has been met
        if i >= stop:
            break
        predictions.append(model.predict_classes(batch,
                                                batch_size=stream.batch_size,
                                                verbose=0))
        j.append(stream.batch_index)
        # Index of the first image in current batch
        begin_index = (stream.batch_index - 1) * stream.batch_size
        # Index of the last image in the current batch
        end_index = (stream.batch_index * stream.batch_size)
        # Files used in current batch
        files = stream.filenames[begin_index:end_index]
        if len(files) != len(predictions[i]):
            print("Number of files and predictions do not match")
            print len('Number of Files: {0}'.format(files))
            print len('Number of Predictions: {0}'.format(predictions[i]))
            break
        for pred, fname in zip(predictions[i], files):
            if pred == 0:
                band_files.append(os.path.join(images, fname))
        bar.update(i)
        i += 1
    print('Images found: {0}'.format(len(band_files)))
    return band_files
    '''


def write2txt(bandPaths):
    '''
    '''
    prefix = bandPaths['01'][0].split('/')[-3]
    with open('/media/jmiller/ubuntuStorage/classified/{0}_bands_classified.txt'.format(prefix), 'w') as txt:
        for imgList in bandPaths.values():
            for path in sorted(imgList):
                txt.write(path + '\n')


def main():
    '''
    '''
    weights_path = '/home/jmiller/Dropbox/school/cnn_stuff/models/040117/040117-1151_paper_model/transVGG.hdf5'
    json_path = '/home/jmiller/Dropbox/school/cnn_stuff/models/040117/040117-1151_paper_model/model_ftil_25_040117-1151.json'
    model = model_from_json(open(json_path, 'r').read())
    model.load_weights(weights_path)
    model.compile(optimizer='SGD',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    base_dir = '/media/jmiller/ubuntuStorage/thesis_images/2015'

    datagen = ImageDataGenerator(rescale=1 / 255.)

    predictions = {}
    for month in sorted(os.listdir(base_dir)):
        predictions[month] = {}
        month_dir = os.path.join(base_dir, month)
        for day in sorted(os.listdir(month_dir)):
            day_dir = os.path.join(month_dir, day)
            predictions[month][day] = classify(model, day_dir)
        # write2txt(predictions)

    with open('./predicted_bands_2014_.json', 'w') as j:
        json.dump(predictions, j, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
