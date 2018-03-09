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
from functools import reduce
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import h5py
import shutil
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime as dt
from glob import glob
from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers import Input
#from keras.layers.core import Flatten, Dense
from keras.layers import Activation, Dropout, Flatten, Dense


def global_average_pooling(x):
    return K.mean(x, axis=(2, 3))


def global_average_pooling_shape(input_shape):
    return input_shape[0:2]


def factors(n):
    return sorted(reduce(list.__add__, ([i, n // i]
                                        for i in range(1, int(n**0.5) + 1) if n % i == 0)))


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


home = os.environ['HOME']
start_time = dt.datetime.strftime(dt.datetime.now(), '%H:%M:%S')
print('Start Time is: {0}'.format(start_time))


##################### Set model parameters ######################
# path to the model weights files.								#
weights_path = ''								#
#
# dimensions of images.											#
img_width, img_height = 256, 256								#
#
# Set directories for training and validation data 				#
val_data_dir = home + '/Dropbox/soni_images/data/val'				#
train_data_dir = home + '/Dropbox/soni_images/data/train'			#
#
# Get ther number of images to determine proper batch size 		#
bandsTrain = glob(train_data_dir + '/bands/*')						#
bandsVal = glob(val_data_dir + '/bands/*')							#
notTrain = glob(train_data_dir + '/not_bands/*')					#
notVal = glob(val_data_dir + '/notbands/*')							#
nb_train_samples = len(bandsTrain) + len(notTrain)					#
nb_val_samples = len(bandsVal) + len(notVal)						#
#
# Determine batch size for train images 						#
trainBatch = factor(nb_train_samples)								#
valBatch = factor(nb_val_samples)									#
print('Train batch size: {0}'.format(trainBatch))				#
print('Validation batch size: {0}'.format(valBatch))			#
#
# Number of epochs to run 										#
nb_epoch = 500													#
#
# The last layer that will not be trained 						#
stop_train = 17													#


img_input = Input(shape=(img_height, img_width, 3))

# build the VGG16 network
model = Sequential()
model.name = "VGGCAM"
model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                 name='block1_conv1', input_shape=(256, 256, 3)))  # 1
model.add(
    Conv2D(
        64,
        (3,
         3),
        activation='relu',
        padding='same',
        name='block1_conv2'))  # 2
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))  # 3

# Block 2
model.add(
    Conv2D(
        128,
        (3,
         3),
        activation='relu',
        padding='same',
        name='block2_conv1'))  # 4
model.add(
    Conv2D(
        128,
        (3,
         3),
        activation='relu',
        padding='same',
        name='block2_conv2'))  # 5
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

# Block 3
model.add(
    Conv2D(
        256,
        (3,
         3),
        activation='relu',
        padding='same',
        name='block3_conv1'))  # 7
model.add(Conv2D(256, (3, 3), activation='relu',
                 padding='same', name='block3_conv2'))
model.add(
    Conv2D(
        256,
        (3,
         3),
        activation='relu',
        padding='same',
        name='block3_conv3'))  # 9
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

# Block 4
model.add(
    Conv2D(
        512,
        (3,
         3),
        activation='relu',
        padding='same',
        name='block4_conv1'))  # 11
model.add(Conv2D(512, (3, 3), activation='relu',
                 padding='same', name='block4_conv2'))
model.add(
    Conv2D(
        512,
        (3,
         3),
        activation='relu',
        padding='same',
        name='block4_conv3'))  # 13
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

# Block 5
model.add(
    Conv2D(
        512,
        (3,
         3),
        activation='relu',
        padding='same',
        name='block5_conv1'))  # 15
model.add(Conv2D(512, (3, 3), activation='relu',
                 padding='same', name='block5_conv2'))
model.add(
    Conv2D(
        512,
        (3,
         3),
        activation='relu',
        padding='same',
        name='block5_conv3'))  # 17

# load the model weights
model.load_weights(weights_path)

# Add GAP layer for CAM
model.add(
    Conv2D(
        2,
        (3,
         3),
        activation='relu',
        padding='same',
        name='gap_conv'))
model.add(AveragePooling2D((16, 16), name='gap'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))


# set the first n layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:stop_train]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
loss_func = 'binary_crossentropy'
lr = 1e-4
momentum = 0.9
model.compile(loss=loss_func,
              optimizer=optimizers.SGD(lr=lr, momentum=momentum),
              metrics=['accuracy'])

# Print model config for documentation
print(model.summary())

print('##### Augmenting Images #####')

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.3,
    rotation_range=0.3,
    vertical_flip=True,
    zoom_range=0.3,
    horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=1. / 255)

# Augment images for training
# bands=0
# not_bands=1
class_mode = 'categorical'
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=trainBatch,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_height, img_width),
    batch_size=valBatch,
    class_mode='binary')

now = dt.datetime.strftime(dt.datetime.now(), '%m%d%y-%H%M')
date = dt.datetime.strftime(dt.datetime.now(), '%m%d%y')

if not os.path.exists(
        home + '/Dropbox/cnn_stuff/models/{0}/{1}'.format(date, now)):
    os.makedirs(home + '/Dropbox/cnn_stuff/models/{0}/{1}'.format(date, now))
prefix = home + '/Dropbox/cnn_stuff/models/{0}/{1}/'.format(date, now)
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
# save model code for reference later
shutil.copyfile(
    home +
    '/Dropbox/cnn_stuff/code/keras_VGGCAM.py',
    prefix +
    'keras_VGGCAM.py')

print('##### Training Model #####')

# fine-tune the model
# verbose (0=no logging; 1=progress bar; 2=one log line per epoch.)
hist = model.fit_generator(
    train_generator,
    verbose=1,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_val_samples,
    callbacks=[early_stopping, checkpointer])

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
    g.write('Learning rate: {0}.\n'.format(lr))
    g.write('Momentum: {0}.\n'.format(momentum))
    g.write('Image dims: {0}X{1}.\n'.format(img_width, img_height))
    g.write('Number of epochs run: {0}.\n'.format(
        len(hist.history['val_loss'])))
    g.write('Batch size for training: {0}\n'.format(trainBatch))
    g.write('Batch size for validation: {0}\n'.format(valBatch))
    end_time = dt.datetime.strftime(dt.datetime.now(), '%H:%M:%S')
    g.write('End Time is: {0}'.format(end_time))

# Plot the loss and accuracy through training
plt.style.use('seaborn-deep')
fig, ax1 = plt.subplots()
fig.set_figheight(16)
fig.set_figwidth(28)
# ax1.set_axis_bgcolor('white')
epochs = range(0, len(hist.history['val_loss']))
pl1 = ax1.plot(
    epochs,
    hist.history['val_loss'],
    color='dodgerblue',
    linewidth=2.75,
    label='Validation Loss')
pl2 = ax1.plot(
    epochs,
    hist.history['loss'],
    color='olive',
    linewidth=2.75,
    label='Training Loss',
    linestyle='dashed')
ax1.set_ylabel('Loss', fontsize=20)
ax1.set_xlabel('Epoch', fontsize=20)
ax1.set_ylim([0., max(hist.history['loss']) + 0.1])
ax1.tick_params(axis='both', labelsize=18)
for spine in ['left', 'right', 'top', 'bottom']:
    ax1.spines[spine].set_color('k')

ax2 = ax1.twinx()
# ax2.set_axis_bgcolor('white')
pl3 = ax2.plot(
    epochs,
    hist.history['val_acc'],
    'orangered',
    linewidth=2.75,
    label='Validation Accuracy')
pl4 = ax2.plot(
    epochs,
    hist.history['acc'],
    'firebrick',
    linewidth=2.75,
    label='Training Accuracy',
    linestyle='dashed')
ax2.set_ylabel('Accuracy', fontsize=20)
ax2.tick_params(axis='both', labelsize=18)
ax2.set_ylim([0., 1.])
ax2.grid(True)
for spine in ['left', 'right', 'top', 'bottom']:
    ax2.spines[spine].set_color('k')
pl = pl1 + pl2 + pl3 + pl4
labs = [a.get_label() for a in pl]
lgd = ax1.legend(
    pl,
    labs,
    bbox_to_anchor=(
        1.05,
        1),
    loc=2,
    borderaxespad=0.,
    fontsize=18)
# lgd=ax1.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.,fontsize=18)
art = []
art.append(lgd)
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
plt.savefig('{0}run_ftil_{1}_{2}.png'.format(prefix, stop_train, now),
            additional_artists=art,
            bbox_inches='tight')
