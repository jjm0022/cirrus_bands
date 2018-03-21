# -*- coding: utf-8 -*-
# @Author: jmiller
# @Date:   2017-05-04 09:25:39
# @Last Modified by:   jmiller
# @Last Modified time: 2018-03-19 20:28:37

import numpy as np
import os
from PIL import Image
from glob import glob
from keras.models import model_from_json
from keras import metrics
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

    datagen = ImageDataGenerator()
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
            gen.n / gen.batch_size),
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
            if ind >= gen.n:
                done = True
                break
        if done:
            break
        bar.update(i)
        i += 1
    print('Images found: {0}'.format(len(band_files)))
    return band_files


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
                  metrics=[metrics.binary_accuracy])
    base_dir = '/media/jmiller/ubuntuStorage/thesis_images/2013_TERRA'


    predictions = {}
    for month in sorted(os.listdir(base_dir)):
        predictions[month] = {}
        month_dir = os.path.join(base_dir, month)
        for day in sorted(os.listdir(month_dir)):
            day_dir = os.path.join(month_dir, day)
            predictions[month][day] = classify(model, day_dir)

    with open('./predicted_bands_2013.json', 'w') as j:
        json.dump(predictions, j, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
