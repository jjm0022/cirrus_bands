# -*- coding: utf-8 -*-
# @Author: jmiller
# @Date:   2018-02-26 19:40:51
# @Last Modified by:   jmiller
# @Last Modified time: 2018-02-26 22:25:02

import os
import shutil
from calendar import month_abbr as months
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

matplotlib.rcParams.update({'font.family': 'Times New Roman'})


def checkDir(path):
    '''
    '''
    if not os.path.exists(path):
        os.makedirs(path)


def get_spot(row, col, file_dict, new_dir):
    '''
    '''
    total_ = 0
    stats = {}
    new_path = os.path.join(new_dir, str(row) + '_' + str(col))
    checkDir(new_path)
    for month in file_dict.keys():
        stats[month] = {}
        month_ = 0
        for day in file_dict[month].keys():
            files = file_dict[month][day]
            for f in files:
                fName = f.split('/')[-1]
                row_ = int(fName.split('_')[6])
                col_ = int(fName.split('_')[7].split('.')[0])
                if row_ == row and col_ == col:
                    total_ += 1
                    month_ += 1
                    f_tmp = os.path.join(
                        '/'.join(f.split('/')[:-2]), day, day, fName)
                    #f_tmp = os.path.join('/media/jmiller/ubuntuStorage/thesis_images/2013', month, day, day, fName)
                    print(f_tmp)
                    print(os.path.exists(f_tmp))
                    shutil.copyfile(f_tmp, os.path.join(new_path, fName))
        stats[month]['total'] = month_
    stats['total'] = total_

    largest_month = []
    m_tmp = 0
    for month, m_value in stats.items():
        try:
            m = int(month)
            m_total = m_value['total']
            if m_total == m_tmp:
                m_tmp = m_total
                largest_month.append(m)
            elif m_total > m_tmp:
                m_tmp = m_total
                largest_month = [m]
            else:
                continue
        except BaseException:
            continue

    print('{0} total images found.'.format(total_))
    for m in largest_month:
        print(
            'Month with the most images: {0} with {1} images'.format(
                months[m], m_tmp))


def getLatLon(row, col):
    '''
    '''
    parallels = np.linspace(90., -81., 20)
    meridians = np.arange(-180., 180, 9)

    lat = parallels[int(row)] - 4.5
    lon = meridians[int(col)] + 4.5
    
    return lat, lon


def getRowCol(lon, lat, topRight=False):
    '''
    '''
    if topRight:
        lat -= 4.5
        lon -= 4.5
    else:
        lat += 4.5
        lon += 4.5
        
    parallels = np.linspace(90. - 4.5, -81. + 4.5, 19)
    meridians = np.arange(-180. + 4.5, 180. - 4.5, 9)

    row = np.where(parallels == lat)[0]
    col = np.where(meridians == lon)[0]
    return (row, col)


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


def getMatrix(dic,
              bounds,
              size=None):
    '''
    bounds: list of the bounds of the map. In the format [n, s, e, w]
    '''
    
    if size is None:
        width = (bounds[2] - bounds[3]) / 9.
        height = (bounds[0] - bounds[1]) / 9.
        size = (int(height), int(width))
    
    
    matrixDict = dict()
    year=None
    
    _, col_ = getRowCol(bounds[-1], bounds[0], topRight=False)
    row_, _ = getRowCol(bounds[-1], bounds[0], topRight=True)
    
    year_matrix = np.zeros(size)
    for month, days in dic.items():
        month_matrix = np.zeros(size)
        for day, images in days.items():
            if not images:
                continue
            for path in images:
                img = path.split('/')[-1]
                if not year:
                    year = img.split('_')[-4].split('-')[0]
                img = path.split('/')[-1]
                row = img.split('_')[6]
                col = img.split('_')[7].split('.')[0]
                lat, lon = getLatLon(row, col)
                if withinArea(lat, lon, bounds):
                    month_matrix[int(int(row) - row_), int(int(col) - col_)] += 1
        year_matrix += month_matrix
        matrixDict[months[int(month)]] = month_matrix
    matrixDict[year] = year_matrix
    return matrixDict


def plotMatrix(matrix, bounds, month, year, ):
    '''
    '''
    print(matrix.max())
    print(np.where(matrix >= 12)[0].shape)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    lats = np.flip(np.arange(bounds[1], bounds[0] + 9.0, 9.), axis=0)
    lons = np.arange(bounds[3], bounds[2] + 9.0, 9.)

    fig = plt.figure()#figsize=(10, 8))
    ax = fig.add_subplot(111)
    m = Basemap(projection='cyl',
                 llcrnrlon=bounds[3], llcrnrlat=bounds[1],
                 urcrnrlon=bounds[2], urcrnrlat=bounds[0],
                 resolution='l', ax=ax)
    xs, ys = m(lons, lats)
    im = ax.pcolormesh(xs, ys, matrix, alpha=0.8,
                       antialiased=True, vmin=0, vmax=1, cmap='jet')
    m.drawcoastlines()
    m.drawmeridians(lons, labels=[True, False, False, True])
    m.drawparallels(lats, labels=[True, False, False, True])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    #plt.tight_layout()
    fig.suptitle('Transverse Band Occurence {0}'.format(year), size=22)
    plot_dir = os.path.join('../heatmaps/localized_views/w_pacific/')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    plt.savefig(os.path.join(plot_dir, 'heatmap_{0}.png'.format(year)), dpi=200)
    plt.clf()
    plt.cla()
    plt.close()


def plotMultipleMatrix(matrix_dict, bounds):
    '''
    '''
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    lats = np.flip(np.arange(bounds[1], bounds[0] + 9.0, 9.), axis=0)
    lons = np.arange(bounds[3], bounds[2] + 9.0, 9.)
    fig = plt.figure(figsize=(6,16)) #subplots(nrows=3, ncols=1, sharex=True, sharey=True)
    
    #plt.gcf().set_size_inches(10, 8)
    ax1 = fig.add_subplot(311)
    m1 = Basemap(projection='cyl',
                 llcrnrlon=bounds[3], llcrnrlat=bounds[1],
                 urcrnrlon=bounds[2], urcrnrlat=bounds[0],
                 resolution='l', ax=ax1)
    xs, ys = m1(lons, lats)
    
    title_size = 22
    small_text_size = 14
    im = ax1.pcolormesh(xs, ys, matrix_dict['2013'] / 45., alpha=0.8,
                            antialiased=True, vmin=0, vmax=1, cmap='jet')
    m1.drawcoastlines()
    m1.drawmeridians(lons)
    m1.drawparallels(lats, labels=[True, False, False, True], size=small_text_size)
    ax1.set_title('2013', size=title_size, fontname='Times New Roman')
    ax1.axis('off')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar1 = fig.colorbar(im, cax=cax, orientation='vertical', fontsize=small_text_size)
    cbar1.set_label(fontsize=small_text_size)
    
    ax2 = fig.add_subplot(312)
    m2 = Basemap(projection='cyl',
                 llcrnrlon=lons[0], llcrnrlat=lats[-1],
                 urcrnrlon=lons[-1], urcrnrlat=lats[0],
                 resolution='l', ax=ax2)
    im = ax2.pcolormesh(xs, ys, matrix_dict['2014'] / 72., alpha=0.8,
             antialiased=True, vmin=0, vmax=1, cmap='jet')
    m2.drawcoastlines()
    m2.drawmeridians(lons)
    m2.drawparallels(lats, labels=[True, False, False, True], size=small_text_size)
    ax2.set_title('2014', size=title_size, fontname='Times New Roman')
    ax2.axis('off')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar2 = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar2.set_label(fontsize=small_text_size)

    ax3 = fig.add_subplot(313)
    m3 = Basemap(projection='cyl',
                 llcrnrlon=lons[0], llcrnrlat=lats[-1],
                 urcrnrlon=lons[-1], urcrnrlat=lats[0],
                 resolution='l', ax=ax3)
    im = ax3.pcolormesh(xs, ys, matrix_dict['2015'] / 80., alpha=0.8,
             antialiased=True, vmin=0, vmax=1, cmap='jet')
    m3.drawcoastlines()
    m3.drawmeridians(lons, labels=[True, False, False, True], size=small_text_size)
    m3.drawparallels(lats, labels=[True, False, False, True], size=small_text_size)
    ax3.set_title('2015', size=title_size, fontname='Times New Roman')
    ax3.axis('off')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar3 = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar3.set_label(fontsize=small_text_size)



    plt.tight_layout()

    plot_dir = os.path.join('../heatmaps/localized_views/w_pacific/')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(os.path.join(plot_dir, 'three_year.png'), bbox_inches='tight', dpi=200)
    plt.clf()
    plt.cla()
    plt.close()

def main():
    '''
        season_dict = {
        'winter': [1, 2, 3],
        'spring': [4, 5, 6],
        'summer': [7, 8, 9],
        'fall': [10, 11, 12]}
    '''
    years = ['2013', '2014', '2015']
    max_ = [45, 72, 80]
    bounds=[27, -27, 171, 108]
    multi_matrix = {}
    for ind, year in enumerate(years):
        print(year)
        json_path = './predicted_bands_{0}.json'.format(year)
        with open(json_path, 'r') as j:
            file_dict = json.load(j)
        matrix_dict = getMatrix(file_dict, bounds)
        multi_matrix[year] = matrix_dict[year]
        plotMatrix(matrix_dict[year] / max_[ind], bounds, month=None, year=year)
    plotMultipleMatrix(multi_matrix, bounds)
    
if __name__ == "__main__":
    main()
