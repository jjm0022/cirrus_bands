# -*- coding: utf-8 -*-
# @Author: jmiller
# @Date:   2018-03-12 20:16:52
# @Last Modified by:   jmiller
# @Last Modified time: 2018-03-21 00:22:21

import numpy as np
import shutil
import os
import json
from PIL import Image
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


def getLatLon(row, col):
    '''
    '''
    parallels = np.fliplr([np.arange(-90., 90. + 9., 9.)])[0]
    meridians = np.arange(-180., 180. + 9., 9.)

    lat = parallels[int(row)] - 4.5
    lon = meridians[int(col)] + 4.5
    return lat, lon


def withinArea(lat, lon, bounds):
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


def plotOverMap(img, lat, lon, new_path):
    '''
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    top = lat + 4.5
    bot = lat - 4.5
    lef = lon - 4.5
    rig = lon + 4.5

    parallels = np.fliplr([np.arange(-90., 90. + 4.5, 4.5)])[0]
    meridians = np.arange(-180., 180. + 4.5, 4.5)

    m = Basemap(projection='cyl',
                lon_0=lon, lat_0=lat,
                llcrnrlon=lef,
                llcrnrlat=bot,
                urcrnrlon=rig,
                urcrnrlat=top,
                resolution='i', ax=ax)

    img_ = Image.open(img)
    img_ = img_.transpose(Image.FLIP_TOP_BOTTOM)
    #img_ = plt.imread(img)
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(parallels, labels=[False, True, True, False])
    m.drawmeridians(meridians, labels=[True, False, False, True])
    m.imshow(img_) #, extent=(lef, rig, bot, top))
    img_name = img.split("/")[-1]
    fig.savefig(os.path.join(new_path, img_name))
    plt.close()



def findHotSpots(dic, bounds, year, current_seasons, location):
    '''
    '''

    seasons = ['', 'winter', 'winter', 'winter',
              'spring', 'spring', 'spring',
              'summer', 'summer', 'summer',
              'fall', 'fall', 'fall']

    for month, days in sorted(dic.items()):
        print('Month: {0}'.format(month))
        for day, images in sorted(days.items()):
            print('\tDay: {0}'.format(day))
            if not images:
                continue
            for path in images:
                if year == '2013':
                    path = os.path.join('/media/jmiller/ubuntuStorage/thesis_images/2013_TERRA/{0}'.format(month), path)
                img = path.split('/')[-1]
                row = img.split('_')[6]
                col = img.split('_')[7].split('.')[0]
                lat, lon = getLatLon(row, col)
                if withinArea(lat, lon, bounds):
                    date = img.split('_')[4]
                    month = date.split('-')[1]
                    season = seasons[int(month)]
                    if season in current_seasons:
                        new_path = '/home/jmiller/Dropbox/images/{0}/{1}/{2}'.format(location, year, season)
                        if not os.path.exists(new_path):
                            os.makedirs(new_path)
                        path = os.path.join('/'.join(path.split('/')[:-1]), day, img)
                        plotOverMap(path, lat, lon, new_path)


def main():
    '''
    [n,s,e,w]
    '''
    years = ['2013', '2014', '2015']
    for year in years:
        print(year)
        json_path = 'predicted_bands_{0}.json'.format(year)
        with open(json_path, 'r') as j:
            dict_ = json.load(j)

        n = [63, 63, -18, 54]
        s = [36, 27, -63, 27]
        e = [36, 36, -27, 171]
        w = [-5, -99, -63, 126]
        locations = ['europe', 'e_us', 's_america', 'east_asia']
        for ind in range(len(locations)):
            print(locations[ind])
            bounds = [n[ind], s[ind], e[ind], w[ind]]

            findHotSpots(dict_, bounds, year, ['spring', 'summer', 'winter', 'fall'], locations[ind])



if __name__ == '__main__':
    main()
