import json
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import font_manager
import numpy as np
import os

matplotlib.rcParams.update({'font.family': 'Times New Roman'})


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
    # monthDict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
    #            7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    season_dict = {
        'winter': [
            1, 2, 3], 'spring': [
            4, 5, 6], 'summer': [
                7, 8, 9], 'fall': [
                    10, 11, 12]}

    if dic:
        total = 0
        year_matrix = np.zeros(size)
        # get the list of months per season
        for season, month_list in season_dict.items():
            # for month, days in dic.items():
            # initialize a matrix of seros for the season
            matrix = np.zeros((size[0], size[1]))
            # iterate through the list of months
            for month in month_list:
                # get the dict of days for each month in the season
                print(dic.keys())
                days = dic['{0:02d}'.format(month)]
                m_total = 0
                #matrix = np.zeros((size[0], size[1]))
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
            # save the season matrix
            matrixDict[season] = matrix
            #matrixDict[monthDict[int(month)]] = matrix
        #matrixDict[year] = year_matrix
        #np.savetxt('{0}_matrix.csv'.format(year), year_matrix, delimiter=',', fmt='%1.1i')
        return matrixDict


def paper_plot(dic):
    '''
    Takes a dic of three year matrices and plots them on a single figure.
    '''
    lats = np.linspace(90., -81, 20)
    lons = np.arange(-180., 180, 9)

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    #years_ = [2013, 2014, 2015]
    #seasons = ['']
    plt.gcf().set_size_inches(10, 8)
    m1 = Basemap(projection='cyl',
                 llcrnrlon=lons[0], llcrnrlat=lats[-1],
                 urcrnrlon=lons[-1], urcrnrlat=lats[0],
                 resolution='l', ax=axes[0, 0])
    xs, ys = m1(np.arange(-180., 180, 9), np.linspace(90., -81, 20))

    max_ = 0
    for key in dic.keys():
        tmp = dic[key].max()
        if tmp > max_:
            max_ = tmp

    title_size = 16
    im = axes[0,
              0].pcolormesh(xs,
                            ys,
                            dic['winter'],
                            alpha=0.8,
                            antialiased=True,
                            vmin=0,
                            vmax=max_,
                            cmap='jet')
    m1.drawcoastlines()
    parallels = np.linspace(90., -81., 20)
    meridians = np.arange(-180., 180, 18)
    axes[0, 0].set_title('Jan. - Mar.', size=title_size,
                         fontname='Times New Roman')
    axes[0, 0].axis('off')
    #cbar0 = fig.colorbar(im, orientation='horizontal', shrink=0.625, ax=axes[0,0], pad=0.02, )
    # cbar0.ax.tick_params(labelsize=10)

    m2 = Basemap(projection='cyl',
                 llcrnrlon=lons[0], llcrnrlat=lats[-1],
                 urcrnrlon=lons[-1], urcrnrlat=lats[0],
                 resolution='l', ax=axes[0, 1])
    im = axes[0,
              1].pcolormesh(xs,
                            ys,
                            dic['spring'],
                            alpha=0.8,
                            antialiased=True,
                            vmin=0,
                            vmax=max_,
                            cmap='jet')
    m2.drawcoastlines()
    axes[0, 1].set_title('Apr. - Jun.', size=title_size,
                         fontname='Times New Roman')
    axes[0, 1].axis('off')
    #cbar1 = fig.colorbar(im, orientation='horizontal', shrink=0.625, ax=axes[0,1], pad=0.02, )
    # cbar1.ax.tick_params(labelsize=10)

    m3 = Basemap(projection='cyl',
                 llcrnrlon=lons[0], llcrnrlat=lats[-1],
                 urcrnrlon=lons[-1], urcrnrlat=lats[0],
                 resolution='l', ax=axes[1, 0])
    im = axes[1,
              0].pcolormesh(xs,
                            ys,
                            dic['summer'],
                            alpha=0.8,
                            antialiased=True,
                            vmin=0,
                            vmax=max_,
                            cmap='jet')
    m3.drawcoastlines()
    axes[1, 0].set_title('Jul. - Sep.', size=title_size,
                         fontname='Times New Roman')
    axes[1, 0].axis('off')
    #cbar2 = fig.colorbar(im, orientation='horizontal', shrink=0.625, ax=axes[1,0], pad=0.02, )
    # cbar2.ax.tick_params(labelsize=10)
    #cbar2.set_label('Number of days with transverse bands', size=14, fontname='Times New Roman')

    m4 = Basemap(projection='cyl',
                 llcrnrlon=lons[0], llcrnrlat=lats[-1],
                 urcrnrlon=lons[-1], urcrnrlat=lats[0],
                 resolution='l', ax=axes[1, 1])
    im = axes[1,
              1].pcolormesh(xs,
                            ys,
                            dic['fall'],
                            alpha=0.8,
                            antialiased=True,
                            vmin=0,
                            vmax=max_,
                            cmap='jet')
    m4.drawcoastlines()
    axes[1, 1].set_title('Oct. - Dec.', size=title_size,
                         fontname='Times New Roman')
    axes[1, 1].axis('off')
    #cbar3 = fig.colorbar(im, orientation='horizontal', shrink=0.625, ax=axes[1,1], pad=0.02, )
    # cbar3.ax.tick_params(labelsize=10)
    #cbar3.set_label('Transverse Band Occurence By Season', size=14, fontname='Times New Roman')

    # fig.subplots_adjust(bottom=0.185)
    cbar_ax = fig.add_axes([0.3, 0.1, 0.4, 0.025])
    # cbar_ax.axis('off')
    plt.tight_layout()
    # pad=0.000001, ) # fraction=0.2,
    cbar = fig.colorbar(im, orientation='horizontal', cax=cbar_ax)
    cbar.set_label('Number of days with transverse bands',
                   size=16, fontname='Times New Roman')
    title = fig.suptitle('Transverse Bands Occurence By Season 2015', size=22)
    title.set_y(0.95)
    fig.subplots_adjust(top=0.875, bottom=0.10, wspace=0.001, hspace=0.001)
    plot_dir = os.path.join(
        '/home/jmiller/Dropbox/school/cnn_stuff/heatmaps/climo/tmp')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(
        os.path.join(
            plot_dir,
            'season_heatmap_2015.png'),
        bbox_inches='tight',
        dpi=200)


if __name__ == '__main__':
    pred_json13 = '/home/jmiller/Dropbox/school/cnn_stuff/code/predicted_bands_2015.json'
    #pred_json14 = '/home/jmiller/Dropbox/school/cnn_stuff/code/predicted_bands_2014.json'
    #pred_json15 = '/home/jmiller/Dropbox/school/cnn_stuff/code/predicted_bands_2015.json'

    with open(pred_json13, 'r') as j:
        pred_dic13 = json.load(j)
    # with open(pred_json14, 'r') as j:
    #    pred_dic14 = json.load(j)
    # with open(pred_json15, 'r') as j:
    #    pred_dic15 = json.load(j)

    matrix13 = map_matrix(pred_dic13,
                          size=(20, 40))
    # matrix14 = map_matrix(pred_dic14,
    #       size=(20, 40))
    # matrix15 = map_matrix(pred_dic15,
    #       size=(20, 40))

    #years = dict()

    #years[2013] = matrix13['2013']
    #years[2014] = matrix14['2014']
    #years[2015] = matrix15['2015']

    paper_plot(matrix13)
