# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 20:35:03 2016

@author: jmiller
"""

import keras.backend as K
from keras.models import model_from_json
from keras import optimizers
import matplotlib.pylab as plt
import numpy as np
import theano.tensor.nnet.abstract_conv as absconv
import cv2
import os
from PIL import Image
from glob import glob
#import imp
#cmaps = imp.load_source('colormaps','/home/jmiller/Dropbox/cnn_stuff/colormap/colormaps.py')
#colorScale='plasma'
#plt.register_cmap(name=colorScale, cmap=cmaps.plasma)
plt.set_cmap('jet')

home=os.environ['HOME']

def get_img(img_path):
  # Load image
  print(img_path)
  original_image = cv2.imread(img_path)
  img=original_image
  if img.shape[0] != 256: 
      img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA )
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  X = img/1.
  X = X.astype('float32')
  X = np.rollaxis(X,2)
  X = np.expand_dims(X,axis=0)
  return X

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

def get_classmap(model, X, nb_classes, batch_size, num_input_channels, ratio):

    inc = model.layers[0].input
    conv6 = model.layers[-4].output
    conv6_resized = absconv.bilinear_upsampling(conv6, ratio,
                                                batch_size=batch_size,
                                                num_input_channels=num_input_channels)
    WT = model.layers[-1].W.T
    conv6_resized = K.reshape(conv6_resized, (-1, num_input_channels, 256 * 256))
    classmap = K.dot(WT, conv6_resized).reshape((-1, nb_classes, 256, 256))
    get_cmap = K.function([inc], classmap)
    return get_cmap([X])
    
def plot_classmap(model,img_path,label,save_path,
                  nb_classes=1,
                  num_input_channels=2,
                  ratio=16):
    """
    Plot class activation map of trained VGGCAM model
    args: VGGCAM_weight_path (str) path to trained keras VGGCAM weights
          img_path (str) path to the image for which we get the activation map
          label (int) label (0 to nb_classes-1) of the class activation map to plot
          nb_classes (int) number of classes
          num_input_channels (int) number of conv filters to add
                                   in before the GAP layer
          ratio (int) upsampling ratio (16 * 14 = 224)
    """
    
    # Load and format data
    #try:
    img=get_img(img_path)
    imgT=load_img(img_path)
    imgT=img_to_array(imgT)
    imgT=imgT.reshape(1,3,256,256)
    #im=img
    img_name=img_path.split('/')[-1].split('.')[0]
    im = cv2.resize(cv2.imread(img_path), (256, 256)).astype(np.float32)
    prediction=model.predict_classes(imgT)
    # Get a copy of the original image
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im_ori = im.copy().astype(np.uint8)
    # VGG model normalisations
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))

    batch_size = 1
    classmap = get_classmap(model,
                            im.reshape(1, 3, 256, 256),
                            nb_classes,
                            batch_size,
                            num_input_channels=num_input_channels,
                            ratio=ratio)

    
    #The visiualization for the first activation
    fig,(ax,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(10,5),dpi=500)
    ax.set_axis_off()
    ax2.set_axis_off()
    fig.tight_layout()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)  
    ax.imshow(im_ori)
    classMap=classmap[0, label, :, :]
    #classMap*=(255.0/classMap.max())
    classMap/=np.max(classMap)    
    cutOff=np.histogram(classMap)[1][6]
    print(cutOff)
    #cam=np.ma.masked_where(classMap>cutOff,classMap)
    cam=classMap
    ax.imshow(cam,
               cmap='jet_r',
               alpha=0.50,
               interpolation='nearest')
    #fig.tight_layout()
    if prediction[0][0]==0:
        pred='Bands'
    else:
        pred='Not Bands'
    ax2.imshow(im_ori)
    ax.set_aspect('auto')
    print(pred)
    #if pred=='Not Bands':
    #    return img_path
    ax2.set_aspect('auto')
    #plt.suptitle('Prediction: {0}'.format(pred))
    #plt.imsave('CAMs/cam.png',cam)
    #ax.set_title('Prediction: {0}'.format(pred))
    #ax2.set_title(pred)
    #plt.colorbar()
    plt.savefig(save_path+'/{0}_cam.png'.format(img_name))
    #plt.show()
    return (cam,im_ori)
    #except Exception as e:
    #    print('Passing: {0}'.format(e))
    #    pass

if __name__ == '__main__':

    import os
    weights_path = '/home/jmiller/Dropbox/school/cnn_stuff/models/040117/040117-1151_paper_model/transVGG.hdf5'
    json_path = '/home/jmiller/Dropbox/school/cnn_stuff/models/040117/040117-1151_paper_model/model_ftil_25_040117-1151.json'
    model = model_from_json(open(json_path, 'r').read())
    model.load_weights(weights_path)
    model.compile(optimizer='SGD',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    img = '/home/jmiller/Dropbox/school/cnn_stuff/images/paper_test_done/gravity_paper2.jpg'

    save_path = '/home/jmiller/Desktop'
    cam=plot_classmap(model,
                  img,
                  0,
                  save_path=save_path,
                  nb_classes=1)


    '''
    preds=[]
    for img in imgList:
        #img2=os.path.join('/home/jmiller/Dropbox/cnn_stuff/images/gibsImages_to_HD/2013/02/05', img)
        #img='/media/jmiller/ubuntu_storage/gibsImages_to_HD_cropped/2013/03/{0:02}/MODIS_Aqua_CorrectedReflectance_TrueColor_2013-03-{1:02}_5_7_6_1.png'.format(day,day)
        cam=plot_classmap(model,img,
                          0,
                          save_path='/home/jmiller/Dropbox/cams/paper',
                          nb_classes=1)

        #if cam == img:
        #    print(img)
        #    break
        try:
            preds.append((cam[1],day))
        except:
            pass
    for pred in preds:
        print(pred)
    '''
    
