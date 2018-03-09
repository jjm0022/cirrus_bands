import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from textwrap import wrap
import matplotlib as mpl
from matplotlib import font_manager
import os
home = os.environ['HOME']


def get_ax(fig, grid, xlabel=None):
    ax = fig.add_subplot(grid, aspect=0.3)
    ax.set_xticks([])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=16, fontname='Times New Roman')
    # ax.imshow(img0)
    ax.set_yticks([])
    return ax


conv2 = 'convolution2d_2_activation.png'
conv14 = 'convolution2d_14_activation.png'
img = 'gibs_11.jpg'


# font_path='/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
#props = font_manager.FontProperties(fname=font_path)
#mpl.rcParams['font.family'] = props.get_name()

fig = plt.figure(figsize=(7, 8))
# labels=['(a) Haze on Jan. 3, 2013',
#		'(b) Haze on Jan. 10, 2013',#
#		'(c) Spatial heatmap for haze events during Jan-Sep, 2013',
#		'(d) Spatial heatmap for haze events during Jan, 2013',
#		'(e) Temporal distrobution of haze events during 2013']

#labels = [ '\n'.join(wrap(l, 30)) for l in labels ]


img0 = mpimg.imread(conv2)
img1 = mpimg.imread(conv14)
img2 = mpimg.imread(img)
# img3=mpimg.imread(dust_persian)
# img4=mpimg.imread(dist)

# master grid
gs_master = gridspec.GridSpec(3, 2, wspace=0.1)

# first row
gs_1 = gridspec.GridSpecFromSubplotSpec(
    2, 2, subplot_spec=gs_master[:2, :])
# first image
ax_1 = get_ax(fig, gs_1[0:])
ax_1.imshow(img0, extent=[0, 1, 0, 1])
# second image
# ax_2=get_ax(fig,gs_1[1])
# ax_2.imshow(img0)


# second row
gs_2 = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_master[2, :])
# first image
ax_3 = get_ax(fig, gs_2[0])
ax_3.imshow(img1, extent=[0, 1, 0, 1])
# second image
ax_4 = get_ax(fig, gs_2[1])
ax_4.imshow(img2, extent=[0, 1, 0, 1])

# third row
# gs_3=gridspec.GridSpecFromSubplotSpec(
#	1,1,subplot_spec=gs_master[2,:],)
# first image
# ax_5=get_ax(fig,gs_3[0],xlabel=labels[4])
# ax_5.imshow(img4)

gs_master.tight_layout(fig)
plt.show()
