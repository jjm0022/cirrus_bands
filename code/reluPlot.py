# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:42:27 2016

@author: jmiller
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)


def ReLU(x):
    return np.maximum(0, x)


X = np.arange(-4, 5, 1)
Y = ReLU(X)

fig, ax = plt.subplots()

ax.axhline(y=0, color='k', lw=0.5)
ax.axvline(x=0, color='k', lw=0.5)
ax.plot(X, Y, 'b', lw=1.0)
ax.set_ylim(-1, 5)
ax.set_title('ReLU')
# ax.grid()
ax.set_xlabel('x', fontsize=22)
ax.set_ylabel('y', fontsize=22)

plt.show()
