# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 20:16:01 2016

@author: jmiller
"""


'''
This is our directory structure:
images/
    train/
        bands/
            bands001.jpg
            bands002.jpg
            ...
        not_bands/
            not_bands001.jpg
            not_bands002.jpg
            ...
    validation/
        bands/
            bands001.jpg
            bands002.jpg
            ...
        not_bands/
            not_bands001.jpg
            not_bands002.jpg
            ...
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime as dt
from glob import glob
from keras import backend as K


def global_average_pooling(x):
    return K.mean(x, axis=(2, 3))


def global_average_pooling_shape(input_shape):
    return input_shape[0:2]


home = os.environ['HOME']
start_time = dt.datetime.strftime(dt.datetime.now(), '%H:%M:%S')
print('Start Time is: {0}'.format(start_time))

# path to the model weights files.
weights_path = 'vgg16_weights.h5'
#top_model_weights_path = 'fc_model.h5'
# dimensions of our images.
img_width, img_height = 256, 256
stop_train = 13

validation_data_dir = home + '/Dropbox/cnn_stuff/images/keras_val'
train_data_dir = home + '/Dropbox/cnn_stuff/images/keras_train'
_bands = glob(train_data_dir + '/bands/*')
_nbands = glob(train_data_dir + '/not_bands/*')
nb_train_samples = len(_bands) + len(_nbands)
nb_validation_samples = 500
nb_epoch = 200
batch_size = 10


# build the VGG16 network
model = Sequential()
model.add(
    ZeroPadding2D(
        (1, 1), input_shape=(
            3, img_width, img_height)))  # layer 0

model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(
    Convolution2D(
        64,
        3,
        3,
        activation='relu',
        name='conv1_2'))  # layer 3
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(
    Convolution2D(
        128,
        3,
        3,
        activation='relu',
        name='conv2_2'))  # layer 7
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(
    Convolution2D(
        256,
        3,
        3,
        activation='relu',
        name='conv3_3'))  # layer 13
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(
    Convolution2D(
        512,
        3,
        3,
        activation='relu',
        name='conv4_3'))  # layer 19
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(
    Convolution2D(
        512,
        3,
        3,
        activation='relu',
        name='conv5_3'))  # layer 25
#model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# load the weights of the VGG16 networks
# (trained on ImageNet, won the ILSVRC competition in 2014)
# note: when there is a complete match between your model definition
# and your weight savefile, you can simply call model.load_weights(filename)
assert os.path.exists(
    weights_path), 'Model weights not found (see "weights_path" variable in script).'
f = h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()
print('Model loaded.')

# Add global average pooling layer
model.add(Lambda(global_average_pooling,
                 output_shape=global_average_pooling_shape,
                 input_shape=model.output_shape[1:]))
act_func = 'sigmoid'
model.add(Dense(2, activation=act_func))

# set the first n layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:stop_train]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
loss_func = 'binary_crossentropy'
model.compile(loss=loss_func,
              optimizer=optimizers.SGD(lr=1e-5, momentum=0.9),
              metrics=['accuracy'])

print('##### Augmenting Images #####')

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    rotation_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=1. / 255)

# Augment images for training
# bands=0
# not_bands=1
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

now = dt.datetime.strftime(dt.datetime.now(), '%m%d%y-%H%M')
date = dt.datetime.strftime(dt.datetime.now(), '%m%d%y')

if not os.path.exists(home + '/Dropbox/cnn_stuff/models/{0}'.format(date)):
    os.makedirs(home + '/Dropbox/cnn_stuff/models/{0}'.format(date))
prefix = home + '/Dropbox/cnn_stuff/models/{0}/'.format(date)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=2,
    mode='auto')
checkpointer = ModelCheckpoint(
    filepath=prefix +
    'transVGG.hdf5',
    verbose=1,
    save_best_only=True)

print('##### Training Model #####')

# fine-tune the model
# verbose (0=no logging; 1=progress bar; 2=one log line per epoch.)
hist = model.fit_generator(
    train_generator,
    verbose=1,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples,
    callbacks=[early_stopping, checkpointer])


# Plot the loss and accuracy through training
plt.style.use('ggplot')
fig, ax1 = plt.subplots()
ax1.set_axis_bgcolor('white')
epochs = range(0, len(hist.history['val_loss']))
ax1.plot(
    epochs,
    hist.history['val_loss'],
    color='dodgerblue',
    linewidth=2.5,
    label='Validation Loss')
ax1.plot(
    epochs,
    hist.history['loss'],
    color='olive',
    linewidth=2,
    label='Training Loss')
ax1.set_ylabel('Loss')
ax1.set_xlabel('Epoch')
for spine in ['left', 'right', 'top', 'bottom']:
    ax1.spines[spine].set_color('k')

ax2 = ax1.twinx()
ax2.set_axis_bgcolor('white')
ax2.plot(
    epochs,
    hist.history['val_acc'],
    'orangered',
    linewidth=2.5,
    label='Validation Accuracy')
ax2.plot(
    epochs,
    hist.history['acc'],
    'firebrick',
    linewidth=2,
    label='Training Accuracy')
ax2.set_ylabel('Accuracy')
for spine in ['left', 'right', 'top', 'bottom']:
    ax2.spines[spine].set_color('k')
ax1.legend(loc=0)
ax2.legend(loc=0)

'''
ax1 = plt.subplot(2,2,1)
ax1.plot(hist.history['val_loss'])
ax1.set_title('val_loss')
ax2 = plt.subplot(2,2,2)
ax2.plot(hist.history['loss'])
ax2.set_title('train_loss')
ax3 = plt.subplot(2,2,3)
ax3.plot(hist.history['val_acc'])
ax3.set_title('val_acc')
ax4 = plt.subplot(2,2,4)
ax4.plot(hist.history['acc'])
ax4.set_title('train_acc')
'''
plt.savefig('{0}run_ftil_{1}_{2}.png'.format(prefix, stop_train, now))


# Save the models weights
model.save_weights(
    '{0}transverseVGG_ftil_{1}_weights_{2}.h5'.format(
        prefix, stop_train, now))

# Save the architecture of the model in a json file
model_json = model.to_json()

f = open('{0}model_ftil_{1}_{2}.json'.format(prefix, stop_train, now), 'w')
f.write(model_json)
f.close()

with open('{0}model_info_{1}.txt'.format(prefix, now), 'w') as g:
    g.write('Model started on {0}.\n'.format(start_time))
    g.write('Number of training samples: {0}.\n'.format(nb_train_samples))
    g.write('Total number of layers {0}.\n'.format(len(model.layers)))
    g.write('Number of layers frozen: {0}.\n'.format(stop_train))
    g.write('Loss function: {0}.\n'.format(loss_func))
    g.write('Image dims: {0}X{1}.\n'.format(img_width, img_height))
    g.write('Number of epochs run: {0}.\n'.format(
        len(hist.history['val_loss'])))
    g.write('Batch size is: {0}.\n'.format(batch_size))
    g.write('The classification function is: {0}.\n'.format(act_func))
    end_time = dt.datetime.strftime(dt.datetime.now(), '%H:%M:%S')
    g.write('End Time is: {0}'.format(end_time))
