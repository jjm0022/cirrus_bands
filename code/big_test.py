# -*- coding: utf-8 -*-
'''
Created on Mon Jul 11 19:30:19 2016

@author: jmiller
'''

#from keras import backend as K
from keras.models import model_from_json
import numpy as np
#import cPickle as pkl
from glob import glob
import cv2
import shutil
from PIL import Image
import os
import sys
import theano
import datetime as dt
import imp
#util = imp.load_source('dl_util','/home/jmiller/Dropbox/cnn_stuff/dl_util.py')
theano.config.exception_verbosity='high'
theano.config.optimizer='fast_compile'

home=os.environ['HOME']
json_path =home+'/Dropbox/cnn_stuff/models/112816/model_ftil_13_112816.json'
weights_path=home+'/Dropbox/cnn_stuff/models/112816/transVGG.hdf5'
#filelist = glob('/home/jmiller/Desktop/test/bands_tmp/*')
#months = glob(home+'/Dropbox/cnn_stuff/images/gibsImages_to_HD/2013/*')
#days=[]
#for month in months:
#    days.append(glob(month+'/*'))

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
        
        
def load_img(path, grayscale=False, target_size=(256,256)):
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


def imgStream(pathLst,num):
    '''
    Takes a list of paths to images and a batch size
    Returns a batch size number of images
    '''
    i=0
    j=num
    # Check to make sure the batch size works with the number of images
        
    if (len(pathLst)/float(num)).is_integer()==False:
        raise NameError('The batch size does not divide into the number of images evenly. Try changing the batch size.')
    while j<(len(pathLst)+1):
        tmp = pathLst[i:j]
        i=j
        j+=num
        yield tmp
        
time = dt.datetime.now()        
batch_size=14

#for month in days:
#    for day in month:
#        images=glob(day+'/*')
images=glob(home+'/Dropbox/cnn_stuff/india/07/*') 
           
#initialize file stream
path_lst = imgStream(images,batch_size)

#load model architecture
model = model_from_json(open(json_path,'r').read())
#load model weights
model.load_weights(weights_path)
#compile model
model.compile(optimizer='SGD',
              loss='binary_crossentropy',
              metrics=['accuracy'])


for lst in path_lst:
  imgs=[]
  paths=[]
  for path in lst:
    img = load_img(path)
    arr=img_to_array(img)
    imgs.append(arr)
    paths.append(path)
  imgs=np.asarray(imgs)
    
  #make predictions on images
  predictions = model.predict_classes(imgs,batch_size=20,verbose=1)
  pred = [None]*len(predictions)
  band_lst=[]
  print(predictions.shape)
  for i in range(predictions.shape[0]):
    if predictions[i]==0:
        band_lst.append(paths[i])
#              else: 
#                  pr[i]=1
  
  #for i in xrange(predictions.shape[0]):
  #    if predictions[i]==1:
  #        pred[i]='Not_bands'
  #    else:
  #        pred[i]='Bands'
  #        guessed_bands.append(lst[i])
  #        tot+=1

  #util.writeTxt('./guesses_bands_{0}.txt'.format(time),lst1=band_lst)
  

#        util.writeTxt('./test_predictions_{0}.txt'.format(time),stat=tot)
#       guessed_bands.pop(0)

  print(len(band_lst))

  for img in band_lst:
    dir_name=img.split('/')[-1]
    #name = band_lst
    shutil.copyfile(img,'{0}/Dropbox/cnn_stuff/images/guessed_bands/{1}'.format(os.getenv('HOME'),dir_name))
    
    
    
    
