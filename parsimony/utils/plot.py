# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 14:13:42 2014

@author: edouard.duchesnay@cea.fr
"""
import numpy as np

def plot_map2d(map2d, plot=None, title=None, limits=None, 
               center_cmap=True):
    import matplotlib.pyplot as plt
    if plot is None:
        plot = plt
    map2d = map2d.squeeze()
    if len(map2d.shape) != 2:
        raise ValueError("input map is not 2D")
    if np.asarray(limits).size is 2:
        mx = limits[0]
        mi = limits[1]
    else:
        mx = map2d.max()
        mi = map2d.min()
    if center_cmap:
        mx = np.abs([mi, mx]).max()
        mi = -mx
    cax = plot.matshow(map2d, cmap=plt.cm.coolwarm)
    frame = plt.gca()
    frame.get_xaxis().set_visible(False)
    frame.get_yaxis().set_visible(False)
    #k = 1
    #while (10 ** k * mx) < 1 and k < 10:
    #    k += 1
    #ticks = np.array([-mi, -mi / 4 - mi / 2, 0, mx / 2, mx / 2,
    #                  mx]).round(k + 2)
    cbar = plt.colorbar(cax)#, ticks=ticks)
    cbar.set_clim(vmin=mi, vmax=mx)
    if title is not None:
        plt.title(title)