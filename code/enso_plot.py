# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 21:25:33 2018

@author: jjmil
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



fig, ax = plt.subplots()
ax.fill_between(l, 0, temps, where=t>0.0, facecolor='red', interpolate=True)
ax.fill_between(l, 0, temps, where=t<0.0, facecolor='blue', interpolate=True)
ax.xaxis.set_major_locator(plt.MultipleLocator(12))
ax.xaxis.set_minor_locator(plt.MultipleLocator(4))



ax.xaxis.set_ticklabels(['','2012','2013','2014','2015','2016','2017', ''],
                        rotation=45)
ax.xaxis.set_ticklabels(['','','May', 'Sep',
                         '','May', 'Sep',
                         '','May', 'Sep',
                         '','May', 'Sep',
                         '','May', 'Sep'], 
                        minor=True,
                        rotation=45)
plt.ylim(-3,3)
