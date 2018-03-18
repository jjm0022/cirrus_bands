# -*- coding: utf-8 -*-
# @Author: jmiller
# @Date:   2018-03-12 20:26:34
# @Last Modified by:   jmiller
# @Last Modified time: 2018-03-12 22:07:35
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import font_manager
import numpy as np
import os

mpl.rcParams.update({'font.family': 'Times New Roman'})


def getLatLon(row, col):
    '''
    '''
    parallels = np.linspace(90., -81., 20)
    meridians = np.arange(-180., 180, 9)

    lat = parallels[int(row)] - 4.5
    lon = meridians[int(col)] + 4.5
    return lat, lon


def map_matrix(dic,
               directory='',
               textFile='',
               size=(20, 40),
               bounds=[-90, 90, -180, 180]):
    year = None
    matrixDict = dict()
    monthDict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    if dic:
        #total = 0
        year_matrix = np.zeros(size)
        for month, days in dic.items():
            #m_total = 0
            matrix = np.zeros((size[0], size[1]))
            for day, images in days.items():
                if not images:
                    continue
                for path in images:
                    img = path.split('/')[-1]
                    if not year:
                        year = img.split('_')[-4].split('-')[0]
                    row = img.split('_')[6]
                    col = img.split('_')[7].split('.')[0]
                    lat, lon = getLatLon(row, col)
                    # if self.withinArea(lat, lon, bounds):
                    coords = (int(row), int(col))
                    matrix[int(coords[0]), int(coords[1])] += 1
            year_matrix += matrix
            # print(year_matrix)
            matrixDict[monthDict[int(month)]] = matrix
        matrixDict[year] = year_matrix
        #np.savetxt('{0}_matrix.csv'.format(year), year_matrix, delimiter=',', fmt='%1.1i')
        return matrixDict


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def paper_plot(dic):
    '''
    Takes a dic of three year matrices and plots them on a single figure.
    '''
    lats = np.linspace(90., -90, 21)
    lons = np.arange(-180., 180 + 9., 9)

    cmap = plt.get_cmap('jet')
    cmap = truncate_colormap(cmap, 0.21, 1.0)

    fig, axes = plt.subplots(3, sharex=True, sharey=True)
    years_ = [2013, 2014, 2015]
    plt.gcf().set_size_inches(8, 12)
    m1 = Basemap(projection='cyl',
                 llcrnrlon=lons[0], llcrnrlat=lats[-1],
                 urcrnrlon=lons[-1], urcrnrlat=lats[0],
                 resolution='l', ax=axes[0])
    xs, ys = m1(np.arange(-180., 180 + 9., 9), np.linspace(90., -90, 21))

    max_ = 0
    for key in dic.keys():
        tmp = dic[key].max()
        if tmp > max_:
            max_ = tmp

    title_size = 16
    im = axes[0].pcolormesh(
        xs,
        ys,
        dic[2013],
        alpha=0.8,
        antialiased=True,
        vmin=0,
        cmap=cmap)
    m1.drawcoastlines()
    parallels = np.linspace(90., -81., 20)
    meridians = np.arange(-180., 180, 18)
    axes[0].set_title('2013', size=title_size, fontname='Times New Roman')
    axes[0].axis('off')
    cbar0 = fig.colorbar(
        im,
        orientation='horizontal',
        shrink=0.625,
        ax=axes[0],
        pad=0.02,
    )
    cbar0.ax.tick_params(labelsize=10)

    m2 = Basemap(projection='cyl',
                 llcrnrlon=lons[0], llcrnrlat=lats[-1],
                 urcrnrlon=lons[-1], urcrnrlat=lats[0],
                 resolution='l', ax=axes[1])
    im = axes[1].pcolormesh(
        xs,
        ys,
        dic[2014],
        alpha=0.8,
        antialiased=True,
        vmin=0,
        cmap=cmap)
    m2.drawcoastlines()
    axes[1].set_title('2014', size=title_size, fontname='Times New Roman')
    axes[1].axis('off')
    cbar1 = fig.colorbar(
        im,
        orientation='horizontal',
        shrink=0.625,
        ax=axes[1],
        pad=0.02,
    )
    cbar1.ax.tick_params(labelsize=10)

    m3 = Basemap(projection='cyl',
                 llcrnrlon=lons[0], llcrnrlat=lats[-1],
                 urcrnrlon=lons[-1], urcrnrlat=lats[0],
                 resolution='l', ax=axes[2])
    im = axes[2].pcolormesh(
        xs,
        ys,
        dic[2015],
        alpha=0.8,
        antialiased=True,
        vmin=0,
        cmap=cmap)
    m3.drawcoastlines()
    axes[2].set_title('2015', size=title_size, fontname='Times New Roman')
    axes[2].axis('off')
    cbar2 = fig.colorbar(
        im,
        orientation='horizontal',
        shrink=0.625,
        ax=axes[2],
        pad=0.02,
    )
    cbar2.ax.tick_params(labelsize=10)
    cbar2.set_label('Number of days with transverse bands',
                    size=14, fontname='Times New Roman')

    # fig.subplots_adjust(bottom=0.185)
    #cbar_ax = fig.add_axes([0.04, 0.15, 0.95, 0.15])
    # cbar_ax.axis('off')
    plt.tight_layout()
    # cbar = fig.colorbar(im, orientation='horizontal', shrink=0.625)# pad=0.000001, ) # fraction=0.2,
    #cbar.set_label('Number of days with transverse bands', size=12, fontname='Times New Roman')
    plot_dir = os.path.join(
        '/Users/jmiller/Dropbox/school/cnn_stuff/heatmaps/climo/paper')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(
        os.path.join(
            plot_dir,
            '3_year_heatmap_.png'),
        bbox_inches='tight',
        dpi=300)


if __name__ == '__main__':
    pred_json13 = '/Users/jmiller/Dropbox/school/cnn_stuff/code/predicted_bands_2013.json'
    pred_json14 = '/Users/jmiller/Dropbox/school/cnn_stuff/code/predicted_bands_2014.json'
    pred_json15 = '/Users/jmiller/Dropbox/school/cnn_stuff/code/predicted_bands_2015.json'

    with open(pred_json13, 'r') as j:
        pred_dic13 = json.load(j)
    with open(pred_json14, 'r') as j:
        pred_dic14 = json.load(j)
    with open(pred_json15, 'r') as j:
        pred_dic15 = json.load(j)

    matrix13 = map_matrix(pred_dic13,
                          size=(20, 40))
    matrix14 = map_matrix(pred_dic14,
                          size=(20, 40))
    matrix15 = map_matrix(pred_dic15,
                          size=(20, 40))

    print(matrix13['2013'].shape)

    years = dict()

    years[2013] = matrix13['2013']
    years[2014] = matrix14['2014']
    years[2015] = matrix15['2015']

    paper_plot(years)
