# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 20:16:01 2016

@author: jmiller
"""


'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
In summary, this is our directory structure:
images/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
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
from keras.layers import Dropout, Flatten, Dense
import datetime as dt


# path to the model weights files.
weights_path = '/home/jmiller/Desktop/vgg16_weights.h5'
#top_model_weights_path = 'fc_model.h5'
# dimensions of our images.
img_width, img_height = 256, 256

validation_data_dir = '/media/jmiller/Ubuntu Storage/training_stuff/binary/keras_val'
train_data_dir = '/media/jmiller/Ubuntu Storage/training_stuff/binary/keras_train'
nb_train_samples = 2048
nb_validation_samples = 500
nb_epoch = 10

# build the VGG16 network
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height))) #layer 0

model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))    #layer 3
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))   #layer 7
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))   #layer 13
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))   #layer 19
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#model.add(ZeroPadding2D((1, 1)))
#model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
#model.add(ZeroPadding2D((1, 1)))
#model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
#model.add(ZeroPadding2D((1, 1)))
#model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))   #layer 25
#model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# load the weights of the VGG16 networks
# (trained on ImageNet, won the ILSVRC competition in 2014)
# note: when there is a complete match between your model definition
# and your weight savefile, you can simply call model.load_weights(filename)
assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
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

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
#top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
model.add(top_model)

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:7]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

print('##### Augmenting Images #####')

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='binary')

print('##### Training Model #####')

# fine-tune the model
hist=model.fit_generator(
        train_generator,
        verbose=1,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)


now = dt.datetime.strftime(dt.datetime.now(), '%Y%m%d-%H_%M_%S')

#Plot the accuracy and loss information
ax1 = plt.subplot(2,2,1)
ax1.plot(hist.history['val_loss'])
ax1.set_title('val_loss')
ax2 = plt.subplot(2,2,2)
ax2.plot(hist.history['loss'])
ax2.set_title('loss')
ax3 = plt.subplot(2,2,3)
ax3.plot(hist.history['val_acc'])
ax3.set_title('val_acc')
ax4 = plt.subplot(2,2,4)
ax4.plot(hist.history['acc'])
ax4.set_title('acc')

plt.savefig('plots/run_{0}.png'.format(now))


# Save the models weights
model.save_weights('trained_models/transverseVGG_weights_{0}.h5'.format(now))

# Save the architecture of the model in a json file
model_json = model.to_json()

f = open('trained_models/model_{0}.json'.format(now), 'w')
f.write(model_json)
f.close()




