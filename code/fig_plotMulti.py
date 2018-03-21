# -*- coding: utf-8 -*-
# @Author: jmiller
# @Date:   2018-03-12 20:32:37
# @Last Modified by:   jmiller
# @Last Modified time: 2018-03-18 21:37:02
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 12:40:36 2018

@author: jjmil
"""
import matplotlib
matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image



class Plot():
    
    
    def __init__(self, shape, figsize=(9, 6)):
        '''
        shape: number of columns and rows (row, col)
        good gridspec tutorial: 
            http://www.sc.eso.org/~bdias/pycoffee/codes/20160407/gridspec_demo.html
        '''
        self.fig = plt.figure(figsize=figsize)
        
        self.gs = gridspec.GridSpec(shape[0], shape[1])
        
        self.shape = shape
        
    def plot_multi(self):
        '''
        '''
        images = ['/media/jmiller/ubuntuStorage/thesis_images/2014/07/24/24/MODIS_Terra_CorrectedReflectance_TrueColor_2014-07-24_5_4_9.jpg',
        '/media/jmiller/ubuntuStorage/thesis_images/2014/07/03/03/MODIS_Terra_CorrectedReflectance_TrueColor_2014-07-03_5_6_11.jpg',
        '/media/jmiller/ubuntuStorage/thesis_images/2015/06/07/07/MODIS_Terra_CorrectedReflectance_TrueColor_2015-06-07_5_5_10.jpg',
        '/media/jmiller/ubuntuStorage/thesis_images/2014/08/20/20/MODIS_Terra_CorrectedReflectance_TrueColor_2014-08-20_5_5_10.jpg',
        '/media/jmiller/ubuntuStorage/thesis_images/2014/04/28/28/MODIS_Terra_CorrectedReflectance_TrueColor_2014-04-28_5_5_11.jpg',
        '/media/jmiller/ubuntuStorage/thesis_images/2013/05/30/30/MODIS_Aqua_CorrectedReflectance_TrueColor_2013-05-30_5_6_10.jpg']
        ind = 0
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                ax = plt.subplot(self.gs[row, col])
                img = Image.open(images[ind]) 
                ax.imshow(img)
                ax.axis('off')
                ind += 1

        # Remove the space between the plots
        self.gs.update(wspace=-0.01, hspace=-0.01)
        
        plot_dir = '../paper_figures/'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plt.savefig(os.path.join(plot_dir, 'e_us.png'),
                    dpi=200,
                    bbox_inches='tight')


def main():
    '''
    '''
    plot = Plot(shape=(2, 3))
    plot.plot_multi()


if __name__ == '__main__':
    main()
