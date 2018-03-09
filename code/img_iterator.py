# -*- coding: utf-8 -*-
# @Author: jmiller
# @Date:   2018-02-25 19:59:13
# @Last Modified by:   jmiller
# @Last Modified time: 2018-02-26 20:02:37
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:03:24 2016

@author: jmiller
"""

import numpy as np
#import re
from six.moves import range
import os
import threading
import json
from glob import glob
from PIL import Image
import progressbar
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from keras import backend as k
import matplotlib
from functools import reduce
matplotlib.rcParams.update({'font.family': 'Times New Roman'})
#from keras.backend import theano_backend as K
home = os.environ['HOME']


def array_to_img(x, dim_ordering='th', scale=True):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'th':
        x = x.transpose(1, 2, 0)
    if scale:
        x += max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return Image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise Exception('Unsupported channel number: ', x.shape[2])


def img_to_array(img, dim_ordering='th'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    if dim_ordering not in ['th', 'tf']:
        raise Exception('Unknown dim_ordering: ', dim_ordering)
    # image has dim_ordering (height, width, channel)
    x = np.asarray(img, dtype='float32')
    if len(x.shape) == 3:
        if dim_ordering == 'th':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if dim_ordering == 'th':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise Exception('Unsupported image shape: ', x.shape)
    return x


def load_img(path, grayscale=False, target_size=None):
    '''Load an image into PIL format.
    # Arguments
      path: path to image file
      grayscale: boolean
      target_size: None (default to original size)
          or (img_height, img_width)
    '''
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    else:  # Ensure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')
    if target_size:
        img = img.resize((target_size[1], target_size[0]))
    return img


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


def move_bad_images(directory, new_location=None, type=None):
    '''
    type: a string that can be 3 values: 'year','day',or 'clean'
        'clean' will only work in one directory and will REMOVE the files that have more than 25% black pixels.
    '''

    if new_location is None:
        home = os.environ['HOME']
        new_location = home + '/Dropbox/cnn_stuff/images/bad_gibs'
    if type == 'year':
        year = glob(directory + '/*')
        months = []
        for month_ in year:
            months.append(glob(month_ + '/*'))
        for month_ in months:
            _month = month_[0].split('/')[-2]
            for day in month_:
                _day = day.split('/')[-1]
                if not os.path.exists(
                        '{0}/{1}/{2}'.format(new_location, _month, _day)):
                    os.makedirs(
                        '{0}/{1}/{2}'.format(new_location, _month, _day))
                for img in glob(day + '/*'):
                    _img = img.split('/')[-1]
                    im = Image.open(img)
                    pixels = im.getdata()
                    black_value = 0
                    nblack = 0
                    for pixel in pixels:
                        tot = 0
                        for pix in pixel:
                            if pix <= black_value:
                                tot += 1
                        if tot == 3:
                            nblack += 1
                    n = len(pixels)
                    if (nblack / float(n)) > 0.25:
                        os.rename(
                            img, '{0}/{1}/{2}/{3}'.format(new_location, _month, _day, _img))

    if type == 'clean':
        num = 0
        print(len(glob(directory + '/*')))
        for img in glob(directory + '/*'):
            im = Image.open(img)
            pixels = im.getdata()
            black_value = 0
            nblack = 0
            for pixel in pixels:
                tot = 0
                for pix in pixel:
                    if pix <= black_value:
                        tot += 1
                if tot == 3:
                    nblack += 1
            n = len(pixels)
            if (nblack / float(n)) > 0.25:
                os.remove(img)
                num += 1
    print('{0} images removed'.format(num))

    if type == 'day':
        num = 0
        _day = directory.split('/')[-1]
        if not os.path.exists('{0}/{1}'.format(new_location, _day)):
            os.makedirs('{0}/{1}'.format(new_location, _day))
        for img in glob(directory + '/*'):
            _img = img.split('/')[-1]
            im = Image.open(img)
            pixels = im.getdata()
            black_value = 0
            nblack = 0
            for pixel in pixels:
                tot = 0
                for pix in pixel:
                    if pix <= black_value:
                        tot += 1
                if tot == 3:
                    nblack += 1
            n = len(pixels)
            if (nblack / float(n)) > 0.33:
                os.rename(img, '{0}/{1}/{2}'.format(new_location, _day, _img))
                num += 1
    print('{0} images moved'.format(num))


def checkBlackSpace(img_array):
    '''
    Checks the image for black pixels
    If more than 30% of the image is black, returns True
    '''
    im = Image.open(img_array)
    pixels = im.getdata()
    black_value = 0
    nblack = 0
    for pixel in pixels:
        tot = 0
        for pix in pixel:
            if pix <= black_value:
                tot += 1
        if tot == 3:
            nblack += 1
    n = len(pixels)
    if (nblack / float(n)) > 0.30:
        return True
    else:
        return False


def cam_check(model, images, class_tag=0, save_path=''):
    '''
    model: keras model object
    images: list of image paths
    class_tag: number representing the tag of a class
    save_path: directory where the cam will be saved
    '''
    import cam_viz as camV

    for img in images:
        cam = camV.plot_classmap(model, img, class_tag, save_path=save_path)


def get_imgs(dir, col, row, quarter=None):
    '''
    Returns a list of images for the given location
    dir: the directory where the images are
    '''
    imgs = []
    with open(dir, 'r') as t:
        for line in t:
            imgs.append(line.strip())
    img_lst = []
    if quarter:
        for img in imgs:
            mon_ = img.split('_')[4].split('-')[1]
            row_ = img.split('_')[6]
            col_ = img.split('_')[7]
            qua_ = img.split('_')[8].split('.')[0]
            if col_ == str(col) and row_ == str(row):
                if qua_ == str(quarter):
                    img_lst.append(img)
    else:
        for img in imgs:
            mon_ = img.split('_')[4].split('-')[1]
            row_ = img.split('_')[6]
            col_ = img.split('_')[7]
            if col_ == col and row_ == row:
                img_lst.append(img)
    return (img_lst, mon_)


class Iterator(object):

    def __init__(self, N, batch_size, shuffle, seed):
        self.N = N
        self.batch_size = batch_size
        self.shuffle = False
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(
            N, batch_size, self.shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        # ensure self.batch_index is 0
        self.reset()
        while True:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(N)
                if shuffle:
                    index_array = np.random.permutation(N)

            current_index = (self.batch_index * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = N - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


class DirectoryIterator(Iterator):

    def __init__(self, directory, image_data_generator,
                 num_classes=1,
                 target_size=(256, 256), color_mode='rgb',
                 dim_ordering='default',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=False, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
        if self.color_mode == 'rgb':
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        if num_classes != 1:
            # first, count the number of samples and classes
            self.nb_sample = 0

            if not classes:
                classes = []
                for subdir in sorted(os.listdir(directory)):
                    if os.path.isdir(os.path.join(directory, subdir)):
                        classes.append(subdir)
            self.nb_class = len(classes)
            self.class_indices = dict(zip(classes, range(len(classes))))

            for subdir in classes:
                subpath = os.path.join(directory, subdir)
                for fname in sorted(os.listdir(subpath)):
                    is_valid = False
                    for extension in white_list_formats:
                        if fname.lower().endswith('.' + extension):
                            is_valid = True
                            break
                    if is_valid:
                        self.nb_sample += 1
                print(
                    'Found %d images belonging to %d classes.' %
                    (self.nb_sample, self.nb_class))

                # second, build an index of the images in the different class
                # subfolders
                self.filenames = []
                self.classes = np.zeros((self.nb_sample,), dtype='int32')
                i = 0
                for subdir in classes:
                    subpath = os.path.join(directory, subdir)
                    for fname in sorted(os.listdir(subpath)):
                        is_valid = False
                        for extension in white_list_formats:
                            if fname.lower().endswith('.' + extension):
                                is_valid = True
                                break
                        if is_valid:
                            self.classes[i] = self.class_indices[subdir]
                            self.filenames.append(os.path.join(subdir, fname))
                            i += 1
            super(
                DirectoryIterator,
                self).__init__(
                self.nb_sample,
                batch_size,
                shuffle,
                seed)

        else:
            # first, count the number of samples
            self.nb_sample = 0
            self.nb_class = 1
            self.no_class = True
            for fname in sorted(os.listdir(directory)):
                is_valid = False
                for extension in white_list_formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    self.nb_sample += 1
            print(
                'Found %d images belonging to %d classes.' %
                (self.nb_sample, self.nb_class))

            # second, build an index of the images
            self.filenames = []
            i = 0
            for fname in sorted(os.listdir(directory)):
                is_valid = False
                for extension in white_list_formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    self.filenames.append(fname)
                    i += 1
            batch_size = factor(self.nb_sample)
            super(
                DirectoryIterator,
                self).__init__(
                self.nb_sample,
                batch_size,
                shuffle,
                seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)
        # The transformation of images is not under thread lock so it can be
        # done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        grayscale = self.color_mode == 'grayscale'
        num = 0
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img_path = os.path.join(self.directory, fname)
            img = load_img(
                img_path,
                grayscale=grayscale,
                target_size=self.target_size)
            x = img_to_array(img, dim_ordering=self.dim_ordering)
            #x = self.image_data_generator.random_transform(x)
            #x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if num > 0:
            print('{0} images skipped.'.format(num))
        return batch_x


class Img_Iterator(object):

    def __init__(self,
                 dim_ordering='th'):
        if dim_ordering == 'th':
            self.channel_index = 1
            self.row_index = 2
            self.col_index = 3
        if dim_ordering == 'tf':
            self.channel_index = 3
            self.row_index = 1
            self.col_index = 2

        self.__dict__.update(locals())

    def flow_from_directory(
            self,
            directory,
            target_size=(
                256,
                256),
            color_mode='rgb',
            classes=None,
            class_mode='binary',
            batch_size=32,
            shuffle=False,
            seed=None,
            save_to_dir=None,
            save_prefix='',
            save_format='jpeg'):
        return DirectoryIterator(
            directory,
            self,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)


class Generator(object):

    def __init__(self, lst=[]):
        self.paths = {}
        self.imgPaths = {}
        if not lst:
            self.lst = []
        else:
            self.lst = lst

    def img_to_array(self, img):
        # image has dim_ordering (height, width, channel)
        x = np.asarray(img, dtype='float32')
        x = x.transpose(2, 0, 1)
        return x

    def load_img(self, path, target_size=(256, 256)):
        '''Load an image into PIL format.
        # Arguments
          path: path to image file
          grayscale: boolean
          target_size: None (default to original size)
              or (img_height, img_width)
        '''
        img = Image.open(path)
        img = img.resize((target_size[1], target_size[0]))
        self.paths[img] = path
        return img

    def generator(self, imgList, batchSize):
        '''
        '''
        start = 0
        while start < len(imgList):
            yield imgList[start:start + batchSize]
            start += batchSize


class Thesis(object):

    def __init__(self):
        self.year = None
        self.month = None
        self.m = False
        self.max = 0
        self.day = None
        self.classified_dir = home + '/Dropbox/cnn_stuff/images/bands_classified/'
        # self.year_matrix=np.zeros((20,40))
        self.matrixDict = dict()
        self.monthDict = {
            1: 'Jan',
            2: 'Feb',
            3: 'Mar',
            4: 'Apr',
            5: 'May',
            6: 'Jun',
            7: 'Jul',
            8: 'Aug',
            9: 'Sep',
            10: 'Oct',
            11: 'Nov',
            12: 'Dec'}

    def check_path(self, fpath):
        '''
        Checks if a path exists. If it does not then it creates it.
        '''
        if not os.path.exists(fpath):
            os.makedirs(fpath)

    def crop(self, fpath, img, height, width, k=0, area=(0, 0, 256, 256)):
        '''
        Crops an image.

        Variables
        -------------------------
        path=path to save the new images
        input=the image path to be cropped
        height=desired crop height
        width=desired crop width
        k=number to distinguish new images
        area=size of desired crop (top,left,bottom,right)
        '''
        import sys
        im = Image.open(img)
        imgwidth, imgheight = im.size
        for i in range(0, imgheight, height):
            for j in range(0, imgwidth, width):
                box = (j, i, j + width, i + height)
                a = im.crop(box)
                try:
                    o = a.crop(area)
                    self.check_path(fpath)
                    newName = fpath + '/' + img.split('/')[-1].split('.')[0]
                    o.save(newName + '_{0}.png'.format(k))
                except BaseException:
                    print(sys.exc_info()[0])
                    pass
                o.close()
                a.close()
                k += 1

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

    def generator(imgList, batchSize):
        '''
        '''
        start = 0
        while start < len(imgList):
            yield imgList[start:start + batchSize]
            start += batchSize

    def classify(self,
                 model,
                 images,
                 batch_size=20,
                 text='',
                 verbose=0,
                 copy=False):

        import shutil
        from multiprocessing import Pool
        from multiprocessing.dummy import Pool as ThreadPool

        pool = ThreadPool()
        tmp = Img_Iterator()
        new_dir = self.classified_dir + \
            '{0}/{1}/'.format(self.year, self.month)
        g = Generator()
        batchSize = factor(len(images))
        gen = g.generator(images, batchSize)
        band_files = []
        for ind, batch in enumerate(gen):
            imgs = pool.map(g.load_img, batch)
            arrays = pool.map(g.img_to_array, imgs)
            arrays = np.asarray(arrays)
            predictions = model.predict_classes(arrays, verbose=0)
            for index, pred in enumerate(predictions):
                if pred == 0:
                    band_files.append(g.paths[imgs[index]])
        print('{0} images found.'.format(len(band_files)))
        if verbose:
            for pred_batch in predictions:
                for pred_ in pred_batch:
                    print pred_
        if text:
            with open(text, 'a') as txt:
                for file in band_files:
                    txt.write(file + '\n')
        if copy:
            for fname in band_files:
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                dir_name = os.path.join(new_dir, fname)
                shutil.copyfile(os.path.join(images, fname), dir_name)

    def getLocation(self, fpath):
        '''
        Parses a image path and returns the image location in a touple
        '''
        month = fpath.split('_')[4].split('-')[1]
        row = fpath.split('_')[6]
        col = fpath.split('_')[7]
        quarter = fpath.split('_')[8].split('.')[0]
        lats = np.linspace(90., -81., 20)
        lons = np.arange(-180., 180., 9)
        lat = lats[int(row)]
        lon = lons[int(col)]
        if quarter == '0':
            pass
        if quarter == '1':
            lon += 4.5
        if quarter == '2':
            lat -= 4.5
        if quarter == '3':
            lat -= 4.5
            lon += 4.5
        newLat = np.linspace(90., -85.5, 40)
        newLon = np.arange(-180., 180, 4.5)
        y = np.where(newLat == lat)[0][0]
        x = np.where(newLon == lon)[0][0]
        return[(y, x), month]

    def map_matrix(self,
                   dic,
                   directory='',
                   textFile='',
                   size=(20, 40),
                   bounds=[-90, 90, -180, 180]):

        if dic:
            total = 0
            year_matrix = np.zeros(size)
            for month, days in dic.items():
                m_total = 0
                matrix = np.zeros((size[0], size[1]))
                for day, images in days.items():
                    if not images:
                        continue
                    for path in images:
                        img = path.split('/')[-1]
                        if not self.year:
                            self.year = img.split('_')[-4].split('-')[0]
                        row = img.split('_')[6]
                        col = img.split('_')[7].split('.')[0]
                        lat, lon = self.getLatLon(row, col)
                        # if self.withinArea(lat, lon, bounds):
                        coords = (int(row), int(col))
                        matrix[int(coords[0]), int(coords[1])] += 1
                year_matrix += matrix
                # print(year_matrix)
                self.matrixDict[self.monthDict[int(month)]] = matrix
                if matrix.max() > self.max:
                    self.max = matrix.max()
            self.matrixDict[self.year] = year_matrix
            np.savetxt(
                '{0}_matrix.csv'.format(
                    self.year),
                year_matrix,
                delimiter=',',
                fmt='%1.1i')
            return self.matrixDict

    def plot_heatmap(self,
                     dic,
                     lats,
                     lons,
                     year=False,
                     custom=False):

        from mpl_toolkits.basemap import Basemap
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        if not self.m:
            self.m = Basemap(projection='cyl',
                             llcrnrlon=lons[0], llcrnrlat=lats[-1],
                             urcrnrlon=lons[-1], urcrnrlat=lats[0],
                             resolution='l')

            # self.m = Basemap(projection='cyl',
            #            llcrnrlon=-135, llcrnrlat=18,
            #            urcrnrlon=-54, urcrnrlat=54,
            #            resolution='l')

        for month, matrix in dic.items():
            if len(month) > 3:
                xs, ys = self.m(np.arange(-180., 180., 9),
                                np.linspace(90., -81., 20))
                #xs,ys = self.m(np.arange(-135., -45., 9),np.linspace(54., 18., 5))
                plt.pcolormesh(
                    xs,
                    ys,
                    matrix,
                    alpha=0.8,
                    antialiased=True,
                    vmax=np.max(matrix),
                    vmin=0)
                cbar = plt.colorbar(
                    orientation='horizontal',
                    shrink=0.625,
                    pad=0.000001)  # fraction=0.2,
                cbar.set_label(
                    'Number of days with transverse bands',
                    size=18,
                    fontname='Times New Roman')
                plt.gcf().set_size_inches(13, 8)
                self.m.drawcoastlines()
                self.m.drawcountries()
                parallels = np.linspace(90., -81., 20)
                meridians = np.arange(-180., 180, 18)
                self.m.drawparallels(
                    parallels, labels=[
                        1, 0, 0, 0], fontsize=6)
                self.m.drawmeridians(
                    meridians, labels=[
                        0, 0, 0, 1], fontsize=6)
                # plt.tight_layout()
                plt.title('Transverse band occurence {0}.'.format(
                    self.year), size=22, fontname='Times New Roman')
                plot_dir = os.path.join(
                    '/home/jmiller/Dropbox/school/cnn_stuff/heatmaps/climo/new/{0}'.format(self.year))
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                plt.savefig(
                    os.path.join(
                        plot_dir,
                        '{0}_heatmap.png'.format(month)))
                plt.close('all')

            else:
                xs, ys = self.m(np.arange(-180., 180, 9),
                                np.linspace(90., -81, 20))
                #xs,ys = self.m(np.arange(-135.,-45.,9),np.linspace(54.,18.,5))
                plt.pcolormesh(
                    xs,
                    ys,
                    matrix,
                    alpha=0.8,
                    antialiased=True,
                    vmax=self.max,
                    vmin=0)
                # plt.pcolormesh(xs,ys,matrix,vmax=np.max(matrix[8:15,11:27]),vmin=0,alpha=0.8,antialiased=True)
                cbar = plt.colorbar(
                    orientation='horizontal',
                    shrink=0.625,
                    pad=0.000001)  # fraction=0.2,
                cbar.set_label(
                    'Number of days with transverse bands',
                    size=18,
                    fontname='Times New Roman')
                plt.gcf().set_size_inches(13, 8)
                self.m.drawcoastlines()
                self.m.drawcountries()
                parallels = np.linspace(90., -81., 20)
                meridians = np.arange(-180., 180, 18)
                self.m.drawparallels(
                    parallels, labels=[
                        1, 0, 0, 0], fontsize=6)
                self.m.drawmeridians(
                    meridians, labels=[
                        0, 0, 0, 1], fontsize=6)
                # plt.tight_layout()
                plt.title('Transverse band occurrence {0} {1}.'.format(
                    month, self.year), size=22, fontname='Times New Roman')
                plot_dir = os.path.join(
                    '/home/jmiller/Dropbox/school/cnn_stuff/heatmaps/climo/new/{0}'.format(self.year))
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                plt.savefig(
                    os.path.join(
                        plot_dir,
                        '{0}_{1}_heatmap.png'.format(
                            month,
                            self.year)))
                plt.close('all')

    def getLatLon(self, row, col):
        '''
        '''
        parallels = np.linspace(90., -81., 20)
        meridians = np.arange(-180., 180, 9)

        lat = parallels[int(row)] - 4.5
        lon = meridians[int(col)] + 4.5
        return lat, lon

    def withinArea(self, lat, lon, bounds):
        '''
        bounds: [n,s,e,w] are the n, s, e, w bounding lat lons
        lat and lon are the coordinates in question
        '''
        left = min(bounds[2:])
        right = max(bounds[2:])
        bottom = min(bounds[:2])
        top = max(bounds[:2])

        if (bottom < lat < top):
            if (left < lon < right):
                return True
        return False

    def findHotSpots(self, dic, bounds):
        '''
        '''
        import calendar
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        import shutil
        tmp_dic = {}
        for month, days in dic.items():
            month_total = 0
            for day, images in days.items():
                if not images:
                    continue
                for path in images:
                    img = path.split('/')[-1]
                    if not self.year:
                        self.year = img.split('_')[-4].split('-')[0]
                    row = img.split('_')[6]
                    col = img.split('_')[7].split('.')[0]
                    lat, lon = self.getLatLon(row, col)
                    if self.withinArea(lat, lon, bounds):
                        new_path = '/home/jmiller/Desktop/{0}/images/{1}_{2}_{3}_{4}'.format(
                            self.year, str(
                                bounds[0]), str(
                                bounds[1]), str(
                                bounds[2]), str(
                                bounds[3]))
                        if not os.path.exists(new_path):
                            os.makedirs(new_path)
                        shutil.copyfile(path, os.path.join(new_path, img))

    def genHistogram(self, dic, bounds):
        '''
        '''
        import calendar
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        tmp_dic = {}
        for month, days in dic.items():
            month_total = 0
            for day, images in days.items():
                if not images:
                    continue
                for path in images:
                    img = path.split('/')[-1]
                    if not self.year:
                        self.year = img.split('_')[-4].split('-')[0]
                    row = img.split('_')[6]
                    col = img.split('_')[7].split('.')[0]
                    lat, lon = self.getLatLon(row, col)
                    if self.withinArea(lat, lon, bounds):
                        month_total += 1
            tmp_dic[calendar.month_abbr[int(month)]] = month_total

        totals = [tmp_dic[x] for x in calendar.month_abbr if x]
        month_abbr = [x for x in calendar.month_abbr if x]
        y_position = np.arange(1, len(month_abbr) + 1)
        print(y_position)
        print(month_abbr)
        fig, ax = plt.subplots()

        plt.bar(y_position, totals,
                align='center',
                color='steelblue',)
        #x_position = [i + 0.5 for i in y_position]
        plt.xlim([y_position[0] - 0.5, y_position[-1] + 0.5])
        plt.xticks(y_position, month_abbr, size=15)
        plt.yticks(size=15)
        plt.tick_params(
            axis='x',
            which='both',
            bottom='on',
            top='off',
            labelbottom='on')
        plt.tick_params(
            axis='y',
            which='both',
            right='off',
            left='off')

        plt.ylabel('Tiles', size=18)
        plt.title('TCB occurence over the United States in {0}'.format(
            self.year), size=22, fontname='Times New Roman')
        plt.tight_layout()
        plot_dir = os.path.join(
            '/home/jmiller/Dropbox/school/cnn_stuff/heatmaps/climo/{0}'.format(self.year))
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(
            os.path.join(
                plot_dir,
                '{0}_histogram.png'.format(
                    self.year)))

    def genHovmoller(self, dic, bounds):
        '''
        '''
        import calendar
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        import scipy
        over_time = None
        for month, days in dic.items():
            month_total = 0
            for day, images in days.items():
                if not images:
                    continue
                day_array = np.zeros((20, 1))
                for path in images:
                    img = path.split('/')[-1]
                    if not self.year:
                        self.year = img.split('_')[-4].split('-')[0]
                    row = img.split('_')[6]
                    col = img.split('_')[7].split('.')[0]
                    lat, lon = self.getLatLon(row, col)
                    if self.withinArea(lat, lon, bounds):
                        day_array[row, 0] += 1
                if not over_time:
                    over_time = day_array
                else:
                    over_time = np.hstack((over_time, day_array))

        y = np.linspace(90., -81., 20)
        y_ticks = np.linspace(90., -81., 10)
        x = np.arange(over_time.shape[1])
        #over_time = scipy.misc.imresize(over_time, (40, over_time.shape[1]), interp='cubic')
        fig, ax = plt.subplots()
        plt.contourf(x, y, over_time, cmap='jet')
        cbar = plt.colorbar(orientation='horizontal',
                            shrink=0.625,
                            pad=0.1,
                            ticks=range(int(over_time.max() + 1)))
        cbar_ticks = [str(i) for i in range(int(over_time.max()) + 1)]
        cbar.ax.set_xticklabels(cbar_ticks, fontname='Times New Roman')
        cbar.set_label('Number of days with transverse bands',
                       size=14, fontname='Times New Roman')
        # plt.gcf().set_size_inches(13,8)
        mon_nums = np.linspace(15, 345, 12)
        month_abbr = [i for i in calendar.month_abbr if i]
        plt.xticks(mon_nums, month_abbr, fontname='Times New Roman')
        plt.yticks(y_ticks, fontname='Times New Roman')
        plt.tick_params(axis='x',
                        which='both',
                        bottom='on',
                        top='off',
                        labelbottom='on')

        plt.tick_params(axis='y',
                        which='both',
                        right='on',
                        left='on')

        plt.ylabel('Latitude in Degrees', fontname='Times New Roman')
        plt.title('TCB occurence over the United States in {0}'.format(
            self.year), size=22, fontname='Times New Roman')
        plt.tight_layout()
        plot_dir = os.path.join(
            '/home/jmiller/Dropbox/school/cnn_stuff/heatmaps/{0}'.format(self.year))
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(
            os.path.join(
                plot_dir,
                '{0}_hovmoller.png'.format(
                    self.year)))


def india(fpath, topLeft=(0, 0), botRight=(0, 0)):
    '''
    Each image is 9 degrees.
    There are 20 rows (0-19) and 40 columns (0-39)
    (0,0) is in top left and (19,39) is in bottom right
    '''
    import shutil
    lst = glob(fpath + '/*')
    print len(lst)
    for iname in lst:
        new_dir = home + '/Dropbox/cnn_stuff/usa'
        fname = iname.split('/')[-1]
        month = iname.split('/')[-2]
        coords = iname.split(
            '/')[-1].split('-')[-1].split('.')[0].split('_')[2:4]
        print int(coords[0])
        if int(coords[0]) >= topLeft[0] and int(coords[0]) <= botRight[0]:
            new_dir = new_dir + '/' + month
            if int(coords[1]) >= topLeft[1] and int(coords[1]) <= botRight[1]:
                print('lookin lookin good')
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                dir_name = os.path.join(new_dir, fname)
                shutil.copyfile(iname, dir_name)


if __name__ == '__main__':

    from keras.models import model_from_json
    import shutil

    home = os.environ['HOME']

    #weights_path = '/home/jmiller/Dropbox/school/cnn_stuff/models/040117/040117-1151_paper_model/transVGG.hdf5'
    #json_path = '/home/jmiller/Dropbox/school/cnn_stuff/models/040117/040117-1151_paper_model/model_ftil_25_040117-1151.json'
    #model = model_from_json(open(json_path, 'r').read())
    # model.load_weights(weights_path)
    # model.compile(optimizer='SGD',
    #              loss='binary_crossentropy',
    #              metrics=['accuracy'])

    fin = Thesis()
    pred_json = '/home/jmiller/Dropbox/school/cnn_stuff/code/predicted_bands_2015.json'
    year = pred_json.split('/')[-1].split('_')[-1].split('.')[0]
    with open(pred_json, 'r') as j:
        pred_dic = json.load(j)

    # matrix=fin.map_matrix(pred_dic,
    #       size=(20,40))

    # fin.plot_heatmap(matrix,
    #         np.linspace(90.,-81,20),
    #         np.arange(-180.,180,9))

    # fin.plot_heatmap(matrix,
    #    np.linspace(54.,18.,4),
    #    np.arange(-135.,-45.,9))

    #fin.genHovmoller(pred_dic, [90,-90,-90,-36])

    fin.genHistogram(pred_dic, [54, 18, -126, -54])

    #bounds = [54,18,-135,-45]

    #fin.findHotSpots(pred_dic, bounds)

    #cam_path = '/home/jmiller/Desktop/{0}/cams/{1}_{2}_{3}_{4}'.format(year, str(bounds[0]), str(bounds[1]), str(bounds[2]), str(bounds[3]))
    # if not os.path.exists(cam_path):
    #    os.makedirs(cam_path)
    #images = sorted(glob('/home/jmiller/Desktop/{0}/images/{1}_{2}_{3}_{4}/*'.format(year, str(bounds[0]), str(bounds[1]), str(bounds[2]), str(bounds[3]))))
    #cam_check(model, images, class_tag=0, save_path=cam_path)
