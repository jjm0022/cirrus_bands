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


def factor(n, max_=50):    
    '''
    Takes number and returns the highest factor that is less than max_.
    '''
    factors = sorted(reduce(list.__add__, ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
    tmp = 1
    for item in factors:
        if item > tmp and item < max_:
            tmp = item
    return tmp


def batch_infer(model, images_path, labels):
    '''
    '''
    imgs_num = sum([len(imgs) for _, _, imgs in os.walk(images_path) is imgs])
    batch_size = factor(imgs_num)
    generator = ImageDataGenerator()
    data_generator = generator.flow_from_directory(images_path, batch_size=batch_size, shuffle=False)
    num_correct = 0
    correct = 0
    for batch in data_generator:
        images = batch[0]
        truth = batch[1]
        truth_tags = [np.where(x == 1.)[0][0] for x in truth]
        tags = []
        predictions = model.predict_on_batch(images)
        for ind, p in enumerate(predictions):
            if np.all(p) == np.all(truth[ind]):
                correct += 1
            tags.append(np.where(p == 1.)[0][0])

        num_correct += sum([1 for x, y in zip(truth_tags, tags) if x == y])

    print(num_correct / imgs_num)
    print(correct / imgs_num)