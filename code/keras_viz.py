# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 18:17:50 2016

@author: jmiller
"""
from __future__ import print_function
import matplotlib
from functools import reduce
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import model_from_json
from keras.models import *
from keras.callbacks import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab as pl
import numpy as np
from glob import glob
import numpy.ma as ma
import cv2
import os
from PIL import Image
import imp
#cmaps = imp.load_source('colormaps','/home/jmiller/Dropbox/cnn_stuff/colormap/colormaps.py')
# colorScale='plasma'
#plt.register_cmap(name=colorScale, cmap=cmaps.plasma)
# plt.set_cmap(cmaps.plasma)


# Define some useful functions for displaying the activations
def nice_imshow(ax, fig, data, vmin=None, vmax=None, cmap='jet'):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = colorScale
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(
        data,
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
        cmap=plt.get_cmap(cmap))
    fig.tight_layout()
    # pl.tight_layout(h_pad=15.0)
    fig.savefig('activations/{0}/{1}_activation.png'.format(dir_name, name))
    #pl.colorbar(im, cax=cax)


def make_mosaic(imgs, nrows, ncols, border=2):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]

    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                           dtype=np.float32)

    paddedh = imshape[0] + border
    # print(paddedh)
    paddedw = imshape[1] + border
    # print(paddedw)
    for i in xrange(nimgs):
        # chan=3
        row = int(np.floor(i / ncols))
        col = i % ncols
        a = row * paddedh
        b = row * paddedh + imshape[0]
        c = col * paddedw
        d = col * paddedw + imshape[1]
        mosaic[a:b,
               c:d] = imgs[i]
    # print(mosaic)
    return mosaic


def factors(n):
    return sorted(reduce(list.__add__, ([i, n // i]
                                        for i in range(1, int(n**0.5) + 1) if n % i == 0)))


def optimal_layout(n):
    f = factors(n)
    tmp2 = 10000
    for i in xrange(1, len(f)):
        if f[-i] * f[-i] == n:
            dims = (f[-i], f[-i])
            break
        for j in xrange(len(f)):
            if (f[-i] * f[j] == n):
                tmp1 = (f[-i] - f[j])
                # print(f[j],f[-i])
            if tmp1 < tmp2 and tmp1 > 0:
                tmp2 = tmp1
                dims = (f[j], f[-i])
    return dims


def get_map(x):
    arr = np.empty(x[0].shape)
    for i in range(x.shape[1]):
        for j in range(x.shape[2]):
            arr[i, j] = np.mean(x[:, i, j])
    arr = cv2.resize(arr, (rows, cols), interpolation=cv2.INTER_AREA)
    return arr


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


def get_img(img_path):
    # Load image
    print(img_path)
    original_image = cv2.imread(img_path)
    img = original_image
    if img.shape[0] != 256:
        img = cv2.resize(img, (rows, cols), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X = img / 1.
    X = X.astype('float32')
    X = np.rollaxis(X, 2)
    X = np.expand_dims(X, axis=0)
    return X, img, original_image


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


def load_img(path, grayscale=False, target_size=(256, 256)):
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


if __name__ == '__main__':
    home = os.environ['HOME']
    json_path = home + '/Dropbox/cnn_stuff/models/101816/model_ftil_13_101816-1641.json'
    weights_path = home + '/Dropbox/cnn_stuff/models/101816/transVGG.hdf5'
    # image_path=home+'/Dropbox/cnn_stuff/images/guessed_bands/*'
    image_path = '/home/jmiller/Desktop/t1.08209.USA3.143.2000m2.jpg'
    # img_lst=glob(image_path)
    # image_path=home+'/Dropbox/cnn_stuff/activations/gibs_36/gibs_36.jpg'
    dir_name = image_path.split('/')[-1].split('.')[0]
    if not os.path.exists('activations/{0}'.format(dir_name)):
        os.makedirs('activations/{0}'.format(dir_name))

    rows, cols = 256, 256

    # Load model
    model = model_from_json(open(json_path, 'r').read())
    model.load_weights(weights_path)

    model.compile(optimizer='SGD',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # the name of the layers we want to visualize (see model definition below)
    layer_name = ['convolution2d_1', 'convolution2d_2', 'convolution2d_3',
                  'convolution2d_4', 'convolution2d_5', 'convolution2d_6',
                  'convolution2d_7', 'convolution2d_8', 'convolution2d_9',
                  'convolution2d_10', 'convolution2d_11', 'convolution2d_12',
                  'convolution2d_13', 'convolution2d_14']
    maps = [None] * len(layer_name)

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    ii = 0
    map_all = np.empty((len(layer_name), rows, cols))

    img = load_img(image_path)
    X = img_to_array(img)
    X = X.reshape((1, X.shape[0], X.shape[1], X.shape[2]))

    for name in layer_name:

        conv_out = layer_dict[name].get_output_at(0)

        inputs = [K.learning_phase()] + model.inputs

        _conv_out1_f = K.function(inputs, [conv_out])

        def conv_out1_f(X):
            # The [0] is to disable the training phase flag
            return _conv_out1_f([0] + [X])

        #pl.imshow(np.squeeze(img), cmap='gray_r')

        # Visualize weights
        #W = model.layers[1].W.get_value(borrow=True)
        #W = np.squeeze(W)
        #print("W shape : ", W.shape)
        #pl.figure(figsize=(15, 15))
        #pl.title('conv1 weights')
        #nice_imshow(pl.gca(), make_mosaic(W, 6, 6), cmap=cm.binary)

        # Visualize convolution result (after activation)
        C1 = conv_out1_f(X)
        C1 = np.squeeze(C1)
        means = []
        for i in range(C1.shape[0]):
            means.append(np.mean(C1[i]))
        ind = np.argpartition(means, -9)[-9:]
        ind2 = np.argpartition(means, -1)[-1:]

        C2 = C1[ind]
        C3 = C1[ind2]

        maps[ii] = get_map(C2)
        singleMap = get_map(C3)
        map_all[ii] = maps[ii]
        pred_label = 'null'

        # The visiualization for the first activation
        fig = pl.figure(figsize=(30, 30))
        #pl.suptitle('{0} Prediction: {1}'.format(name,pred_label))
        dims = optimal_layout(C2.shape[0])
        ax = pl.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        nice_imshow(ax, fig, make_mosaic(C2, dims[0], dims[1]))
        pl.close()

        plt.imsave(
            'activations/{0}/{1}_highest.png'.format(dir_name, name), singleMap)
        # fig.savefig('activations/{0}_activation.png'.format(name))
        #plt = fig.gca()
        # plt.title
        # plt.tight_layout()
        #ax.imsave('activations/{0}_activation.png'.format(name),make_mosaic(C1z,dims[0],dims[1]),cmap ='jet')
        # pl.imsave('activations/{0}/{1}'.format(dir_name,image_path.split('/')[-1]),img)
        # plt.imsave('activations/{0}/{1}_map.png'.format(dir_name,name),C1)
        # plt.tight_layout()
        #map_all= np
        ii += 1

    total_map = get_map(map_all)
    plt.imsave('activations/{0}/avg_map.png'.format(dir_name, name), total_map)


'''
  class_weights=model.layers[-1].get_weights()[0]
  final_conv_layer = get_output_layer(model, "conv5_3")
  get_output = K.function([model.layers[0].input], [final_conv_layer.output,model.layers[-1].get_output_at(0)])

  for img_path in img_lst:
    dir_name=img_path.split('/')[-1].split('.')[0]
    X,img,original_image=get_img(img_path)
    [conv_outputs, predictions] = get_output([X])
    conv_outputs = conv_outputs[0, :, :, :]

    prediction = model.predict_classes(X)
    if prediction==0:
        pred_label = 'bands'

    cam=np.zeros(dtype=np.float32,shape=conv_outputs.shape[1:3])
    for i,w in enumerate(class_weights[:,0]):
            cam+=w * conv_outputs[i,:,:]
    print('predictions: {0} '.format(predictions))
    cam /=np.max(cam)
    width, height, _ = original_image.shape
    cam=cv2.resize(cam, (height,width))
    heatmap=cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam>0.2)] = 0
    img = heatmap*.5 + original_image
    cv2.imwrite('CAMs/{0}CAM.png'.format(dir_name),img)



'''
