import numpy as np
import os
from glob import glob
from PIL import Image
home = os.environ['HOME']


def load_img(path, grayscale=False, target_size=False):
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

    directory = home + '/Dropbox/cnn_stuff/dmc_images/val/not_bands/*'
    img_lst = glob(directory)
    i = 1
    category = 'not_bands'
    for img in img_lst:
        new_img = load_img(img)
        new_img.save(
            home +
            '/Dropbox/cnn_stuff/dmc_images/test/not_bands/' +
            '{0}_test_{1:03d}.jpg'.format(
                category,
                i))
        i += 1
