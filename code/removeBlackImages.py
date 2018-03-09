# -*- coding: utf-8 -*-
# @Author: jmiller
# @Date:   2017-05-02 09:34:56
# @Last Modified by:   jmiller
# @Last Modified time: 2017-05-02 11:13:41




import os
from glob import glob
from PIL import Image
import multiprocessing
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool


def checkDir(path):
    '''
    '''
    if not os.path.exists(path):
        os.makedirs(path)


def move(result):
    '''
    '''
    new_location = '/media/jmiller/ubuntuStorage/thesis_images/bad_images'
    if result[1] is True:
        month = result[0].split('/')[6]
        day = result[0].split('/')[7]
        year = result[0].split('/')[5]
        img = result[0].split('/')[-1]
        #print(img)
        os.rename(result[0], os.path.join(new_location, year, month, day, img))
        return 1
    else:
    	return 0


def dummy(imgList):
    '''
    '''
    p = ThreadPool(processes=multiprocessing.cpu_count())
    results = p.map(checkBlackSpace, imgList, chunksize=10)
    nums = p.map(move, results)
    num=sum(nums)
    p.close()
    return num


def checkBlackSpace(img):
    '''
    Checks the image for black pixels
    If more than 30% of the image is black, returns True
    '''
    im = Image.open(img)
    pixels = im.getdata()
    black_value = 0
    nblack = 0
    for pixel in pixels:
        tot = 0
        for pix in pixel:    
            if pix <= black_value:
                tot += 1
        if tot == 3:                    
            nblack += 1
    n = len(pixels)
    if (nblack / float(n)) > 0.30:
        return (img, True)
    else:
        return (img, False)


def main():
    '''
    '''
    workingDir = '/media/jmiller/ubuntuStorage/thesis_images/2015'
    new_location = '/media/jmiller/ubuntuStorage/thesis_images/bad_images'
    year = str(2015)
    months = sorted(os.listdir(workingDir))
    for month in months:
        print('Month: {0}'.format(month))
        monthsDir = os.path.join(workingDir, month)
        days = sorted(os.listdir(monthsDir))
        for day in days:
            print('\tDay: {0}\r'.format(day))
            checkDir(os.path.join(new_location, year, month, day))
            dayDir = os.path.join(monthsDir, day)
            img_list = glob(dayDir + '/*')
            num = dummy(img_list)
            print('\t{0} images moved in {1}'.format(num, day))


if __name__ == '__main__':
    main()



