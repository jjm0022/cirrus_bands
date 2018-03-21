# -*- coding: utf-8 -*-
# @Author: jmiller
# @Date:   2018-02-26 19:40:51
# @Last Modified by:   J.J. Miller
# @Last Modified time: 2018-03-20 14:28:23

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from plot_heatmaps import Plot

matplotlib.rcParams.update({'font.family': 'Times New Roman'})


def checkDir(path):
    '''
    '''
    if not os.path.exists(path):
        os.makedirs(path)


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


def getSeasonTotals(dic):
    '''
    
    '''
    from calendar import month_abbr as months

    season_dict = {'Winter': [1, 2, 3],
                   'Spring': [4, 5, 6],
                   'Summer': [7, 8, 9],
                   'Fall': [10, 11, 12]}
    # Loop through each year
    for year, values in dic.items():
        seasons = dict()
        # Get the totals for each season
        for season, inds in season_dict.items():
            # Create a blank matrix of the approptiate size
            m = np.zeros(dic[year][year].shape)
            # Get the totals for the season
            for ind in inds:
                m += values[months[ind]]
            seasons[season] = m
        # store the totals in the dict that was provided
        for season, value in seasons.items():
            dic[year][season] = value 
    return dic        
    

def getMatrix(dic,
              bounds,
              size=None):
    '''
    bounds: list of the bounds of the map. In the format [n, s, e, w]
    '''
    from calendar import month_abbr as months
    
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


def plotMatrix(matrix, bounds, year, month=None, season=None ):
    '''
    If season and month are provided at the same time, nothing will be plotted
    '''
    # Helps with colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    if season and month:
        print("Please only provide a season OR a month. Not both")
        return 
    
    lats = np.flip(np.arange(bounds[1], bounds[0] + 9.0, 9.), axis=0)
    lons = np.arange(bounds[3], bounds[2] + 9.0, 9.)

    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot(111)
    m = Basemap(projection='cyl',
                 llcrnrlon=bounds[3], llcrnrlat=bounds[1],
                 urcrnrlon=bounds[2], urcrnrlat=bounds[0],
                 resolution='l', ax=ax)
    xs, ys = m(lons, lats)
    if matrix.max() <= 1.:
        vmax = 1.0
    else:
        vmax = matrix.max()
    im = ax.pcolormesh(xs, ys, matrix, alpha=0.8,
                       antialiased=True, vmin=0, vmax=vmax, cmap='jet')
    m.drawcoastlines()
    # Re-calculate the lats and lons so only every other lon is shown
    lats = np.flip(np.arange(bounds[1], bounds[0] + 9.0, 9.), axis=0)
    lons = np.arange(bounds[3], bounds[2] + 18.0, 18.)
    
    m.drawmeridians(lons, labels=[False, False, False, True], fontsize=12)
    m.drawparallels(lats, labels=[True, False, False, False], fontsize=12)
    
    # Set up color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=14)

    # Set up titles and file names depending on year and month/season
    # This also accounts for average plots
    if year == 'average':
        if month:
            title = 'Transverse Band Occurence {0} 3 Year Average'.format(month)
            fileName = 'heatmap_{0}_{1}.png'.format(month, year)
            path = '../heatmaps/climo/{0}/{1}'.format(year, 'months')
        elif season:
            title = 'Transverse Band Occurence {0} 3 Year Average'.format(season)
            fileName = 'heatmap_{0}_{1}.png'.format(season, year)
            path = '../heatmaps/climo/{0}/{1}'.format(year, 'seasons')
        elif not season and not month:
            title = 'Transverse Band Occurence 3 Year Average'
            fileName = 'heatmap_{0}.png'.format(year)
            path = '../heatmaps/climo/{0}'.format(year)
    else:
        if month:
            title = 'Transverse Band Occurence {0} {1}'.format(month, year)
            fileName = 'heatmap_{0}_{1}.png'.format(month, year)
            path = '../heatmaps/climo/{0}/{1}'.format(year, 'months')
        elif season:
            title = 'Transverse Band Occurence {0} {1}'.format(season, year)
            fileName = 'heatmap_{0}_{1}.png'.format(season, year)
            path = '../heatmaps/climo/{0}/{1}'.format(year, 'seasons')
        elif not season and not month:
            title = 'Transverse Band Occurence {0}'.format(year)
            fileName = 'heatmap_{0}.png'.format(year)
            path = '../heatmaps/climo/{0}'.format(year)
    
    fig.suptitle(title, size=22)
    plot_dir = os.path.join(path)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    plt.savefig(os.path.join(plot_dir, fileName),
                dpi=200,
                bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()

def main():
    '''

    '''
    from calendar import month_abbr as months
    from calendar import month_name
    years = ['2013', '2014', '2015']
    bounds=[90, -90, 180, -180]
    multi_matrix = {}
    for ind, year in enumerate(years):
        print(year)
        json_path = './predicted_bands_{0}.json'.format(year)
        with open(json_path, 'r') as j:
            file_dict = json.load(j)
        matrix_dict = getMatrix(file_dict, bounds)
        multi_matrix[year] = matrix_dict
        plotMatrix(matrix_dict[year], bounds, month=None, year=year)

    # Get the avreage over the three years
    average = np.zeros(multi_matrix['2013']['2013'].shape)
    for year in years:
        average += multi_matrix[year][year]
    average /= float(len(years))
    plotMatrix(average, bounds, year='average')

    ## Plot the season totals for each year.
    multi_matrix = getSeasonTotals(multi_matrix)
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']    
    for year in years:
        for season in seasons:
            plotMatrix(multi_matrix[year][season], bounds, year, season=season)
            
    # Plot the Season averages
    for season in seasons:
        average = np.zeros(multi_matrix['2013']['2013'].shape)
        for year in years:
            average += multi_matrix[year][season]
        average /= float(len(years))
        multi_matrix[season] = average
    plot = Plot(bounds, (2,3))
    plot.plot_multi(multi_matrix)
    
    # Plot the monthly totals for each year.  
    for year in years:
        for ind in range(1,13):
            month = months[ind]
            plotMatrix(multi_matrix[year][month], bounds, year, month=month_name[ind])

    # Plot the monthly averages
    for ind in range(1, 13):
        month = months[ind]
        average = np.zeros(multi_matrix['2013']['2013'].shape)
        for year in years:
            average += multi_matrix[year][month]
        average /= float(len(years))
        plotMatrix(average, bounds, year='average', month=month_name[ind])
    
if __name__ == "__main__":
    main()
