# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 14:13:42 2014

@author: edouard.duchesnay@cea.fr
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_map2d(map2d, plot=None, title=None):
    if plot is None:
        plot = plt
    map2d = map2d.squeeze()
    if len(map2d.shape) != 2:
        raise ValueError("input map is not 2D")
    cax = plot.matshow(map2d, cmap=plt.cm.coolwarm)
    frame = plt.gca()
    frame.get_xaxis().set_visible(False)
    frame.get_yaxis().set_visible(False)
    mx = np.abs(map2d).max()
    k = 1
    while (10 ** k * mx) < 1 and k < 10:
        k += 1
    ticks = np.array([-mx, -mx / 4 - mx / 2, 0, mx / 2, mx / 2,
                      mx]).round(k + 2)
    cbar = plt.colorbar(cax, ticks=ticks)
    cbar.set_clim(vmin=-mx, vmax=mx)
    if title is not None:
        plt.title(title)