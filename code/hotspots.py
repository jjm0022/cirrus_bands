# -*- coding: utf-8 -*-
# @Author: jmiller
# @Date:   2018-02-26 19:40:51
# @Last Modified by:   jmiller
# @Last Modified time: 2018-02-26 22:25:02

import os
import shutil
from calendar import month_abbr as months
import json


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


def findHotSpots(self, dic, bounds):
    '''
    '''
    import calendar
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import shutil
    tmp_dic = {}
    for month, days in dic.items():
        month_total = 0
        for day, images in days.items():
            if not images:
                continue
            for path in images:
                img = path.split('/')[-1]
                if not self.year:
                    self.year = img.split('_')[-4].split('-')[0]
                row = img.split('_')[6]
                col = img.split('_')[7].split('.')[0]
                lat, lon = self.getLatLon(row, col)
                if self.withinArea(lat, lon, bounds):
                    new_path = '/home/jmiller/Desktop/{0}/images/{1}_{2}_{3}_{4}'.format(
                        self.year, str(
                            bounds[0]), str(
                            bounds[1]), str(
                            bounds[2]), str(
                            bounds[3]))
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)
                    shutil.copyfile(path, os.path.join(new_path, img))


def main():
    '''
    '''
    json_path = './predicted_bands_2014.json'
    with open(json_path, 'r') as j:
        file_dict = json.load(j)
    get_spot(15, 12, file_dict, '/home/jmiller/Desktop/hotspots/')


if __name__ == "__main__":
    main()
