#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2021

import copy, sys, code # code.interact(local=locals())
import numpy as np
import scipy.special as sp

import matplotlib.pyplot as plt
from matplotlib import rcParams, rc
import matplotlib.ticker as mticker
import cartopy.crs as ccrs

def discretize_fabric(nlm, lm, latres=60):

    #latres = 60 # latitude resolution on S^2        
    theta = np.linspace(0,   np.pi,   latres) # CO-LAT 
    phi   = np.linspace(0, 2*np.pi, 2*latres) # LON
    phi, theta = np.meshgrid(phi, theta) # gridded 
    lon, colat = phi, theta
    lat = np.pi/2-colat
    
    _,nlm_len = lm.shape
    F = np.real(np.sum([ nlm[ii]*sp.sph_harm(lm[1][ii], lm[0][ii], phi,theta) for ii in np.arange(nlm_len) ], axis=0))
    
    return (F, lon,lat)

def plot_ODF(nlm, lm, ax=None, cmap='PuOr_r', cblbl=''):
    
    F, lon,lat = discretize_fabric(nlm, lm)
    
    if lm.shape[1]>5: # full ODF
        lvls = np.linspace(0.0,0.5,6) # Contour lvls
        Fplot = F
        cmap = 'Greys'
    
    else: # n2m (dipole) contribution
        lvlmax = 0.7
        lvls = np.linspace(-lvlmax,lvlmax,9) # Contour lvls
        Fplot = F
    
    hdistr = ax.contourf(np.rad2deg(lon), np.rad2deg(lat), Fplot, transform=ccrs.PlateCarree(), levels=lvls, extend='max', cmap=cmap)

    kwargs_gridlines = {'ylocs':np.arange(-90,90+30,30), 'xlocs':np.arange(0,360+45,45), 'linewidth':0.5, 'color':'black', 'alpha':0.25, 'linestyle':'-'}
    gl = ax.gridlines(crs=ccrs.PlateCarree(), **kwargs_gridlines)
    gl.xlocator = mticker.FixedLocator(np.array([-135, -90, -45, 0, 90, 45, 135, 180]))

    cb1 = plt.colorbar(hdistr, ax=ax, fraction=0.075, aspect=9,  orientation='horizontal', pad=0.1, ticks=lvls[::4])   
    cb1.set_label(cblbl)
    cb1.ax.xaxis.set_ticks(lvls, minor=True)
