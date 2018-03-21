# -*- coding: utf-8 -*-
# @Author: jmiller
# @Date:   2017-05-02 09:34:56
# @Last Modified by:   jmiller
# @Last Modified time: 2018-03-19 18:19:24


import os
from PIL import Image
import threading
from queue import Queue
import logging


class Checker(threading.Thread):
    def __init__(self, queue):
        super(Checker, self).__init__()
        self.queue = queue
        self.total_bad_images = 0
        self.months = list()

    def run(self):
        while True:
            img = self.queue.get()
            if not os.path.exists(img):
                print('Does not exist: {0}'.format(img))
            month = img.split('/')[6]
            if month not in self.months:
                print(month)
                self.months.append(month)
            try:
                result = checkBlackSpace(img)
                self.total_bad_images += move(result)
            except Exception as e:
                logging.warn(e)
            self.queue.task_done()


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
        checkDir(os.path.join(new_location, year, month, day))
        os.rename(result[0], os.path.join(new_location, year, month, day, img))
        return 1
    else:
        return 0


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
    q = Queue()
    threads = list()
    num_threads = 10

    for i in range(num_threads):
        threads.append(Checker(q))
        threads[-1].start()

    workingDir = '/media/jmiller/ubuntuStorage/thesis_images/2013_TERRA'
    new_location = '/media/jmiller/ubuntuStorage/thesis_images/bad_images'
    year = str(2013)
    months = sorted(os.listdir(workingDir))
    for month in months:
        monthsDir = os.path.join(workingDir, month)
        days = sorted(os.listdir(monthsDir))
        for day in days:
            checkDir(os.path.join(new_location, year, month, day))
            dayDir = os.path.join(monthsDir, day, day)
            for img in os.listdir(dayDir):
                img_path = os.path.join(dayDir, img)
                q.put(img_path)


if __name__ == '__main__':
    main()
