
import pycurl
from glob import glob
import os
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool


def getImg(c, url, prefix):
    c.setopt(c.URL, url)
    layer_name = "MODIS_Terra_CorrectedReflectance_TrueColor"
    level = str(5)
    date = url.split('/')[8]
    col = url.split('/')[11]
    row = url.split('/')[12].split('.')[0]
    with open('{0}_{1}_{2}_{3}_{4}.jpg'.format(layer_name, date, level, col, row) 'w') as f:
        c.setopt(c.WRITEFUNCTION, f.write)
        c.perform()


https: // gibs.earthdata.nasa.gov / wmts / epsg4326 / best / MODIS_Terra_CorrectedReflectance_TrueColor / default / 2014 - 01 - 01 / 250m / 5 / 8 / 20.jpg

pool = ThreadPool()

LAYER_NAME = "MODIS_Terra_CorrectedReflectance_TrueColor"
FORMAT = "jpg"

fileBase = '/media/jmiller/ubuntu_storage/thesis_images'

with pycurl.Curl() as c:
    for year in range(2014, 2016):
    	for month in range(1, 13):
    		month = str(month).zfill(2)
            files = sorted(glob(os.path.join(fileBase, month, '*.txt'))
    		for ind, day in enumerate(range(1, 31)):
                day = str(day).zfill(2)
                date = year + '-' + month + '-' + day
                filePath = fileBase + '/' + year + '/' + month + '/' + day +'/'
                if date + '.txt' in files:
                    with open(date + '.txt', 'r') as txt:
                        lines = f.read().splitlines()
                        for line in lines:
                            print(line)

