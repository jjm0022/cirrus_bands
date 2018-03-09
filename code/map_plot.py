# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 18:17:50 2016

@author: jmiller
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from glob import glob
import os

home = os.environ['HOME']


directory = home + '/Dropbox/cnn_stuff/images/gibsImages_to_HD/2013/03/10/*'
images = sorted(glob(directory))

lats = np.linspace(90., -81., 20)
lons = np.arange(-180., 180., 9)

m = Basemap(projection='cyl',
            llcrnrlon=min(lons), llcrnrlat=min(lats),
            urcrnrlon=max(lons), urcrnrlat=max(lats),
            resolution='c')

for image in images:
    coords = image.split('/')[-1].split('-')[-1].split('.')[0].split('_')[2:4]

    upper_lat = lats[int(coords[0])]
    left_lon = lons[int(coords[1])]
    lower_lat = lats[int(coords[0])] - 9
    right_lon = lons[int(coords[1])] + 9

    plt.imshow(
        plt.imread(image),
        extent=(
            left_lon,
            right_lon,
            lower_lat,
            upper_lat))

m.drawcoastlines()
m.drawcountries()
plt.show()
