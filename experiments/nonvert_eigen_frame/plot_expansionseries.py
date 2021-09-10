#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2021

import numpy as np
import copy, sys, os, code # code.interact(local=locals())
sys.path.insert(0, '../../lib')

from fabrictools import *
import scipy.special as sp

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams, rc
import matplotlib.ticker as mticker
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cartopy.crs as ccrs
import seaborn as sns

#--------------------

FS = 13
rc('font',**{'family':'serif','sans-serif':['Times'],'size':FS})
rc('text', usetex=True)
rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{physics} \usepackage{txfonts} \usepackage{siunitx}'

#--------------------

def plot_Ylm(F,lon,lat, ax=None, tlbl='', lbl='', cmap='PuOr_r'):
    
    lvlmax = 0.7
    lvls = np.linspace(-lvlmax,lvlmax,9) # Contour lvls
    Fplot = F
    
    hdistr = ax.contourf(np.rad2deg(lon), np.rad2deg(lat), Fplot, transform=ccrs.PlateCarree(), levels=lvls, extend='both', cmap=cmap)

    kwargs_gridlines = {'ylocs':np.arange(-90,90+30,30), 'xlocs':np.arange(0,360+45,45), 'linewidth':0.5, 'color':'black', 'alpha':0.25, 'linestyle':'-'}
    gl = ax.gridlines(crs=ccrs.PlateCarree(), **kwargs_gridlines)
    gl.xlocator = mticker.FixedLocator(np.array([-135, -90, -45, 0, 90, 45, 135, 180]))

    cb1 = plt.colorbar(hdistr, ax=ax, fraction=0.075, aspect=9.5,  orientation='horizontal', pad=0.1, ticks=lvls[::4])   
    cb1.set_label(lbl)
    ax.set_title(tlbl, pad=10, fontsize=FS)
    cb1.ax.xaxis.set_ticks(lvls, minor=True)
    
def discretize_fabric(nlm, lm, latres=60):

    #latres = 60 # latitude resolution on S^2        
    theta = np.linspace(0,   np.pi,   latres) # CO-LAT 
    phi   = np.linspace(0, 2*np.pi, 2*latres) # LON
    phi, theta = np.meshgrid(phi, theta) # gridded 
    lon, colat = phi, theta
    lat = np.pi/2-colat
    
    _,nlm_len = lm.shape
    F = np.sum([ nlm[ii]*sp.sph_harm(lm[1][ii], lm[0][ii], phi,theta) for ii in np.arange(nlm_len) ], axis=0)
    
    return (F, lon,lat)

lw = 1.6
legkwargs = {'frameon':True, 'fancybox':False, 'edgecolor':'k', 'framealpha':0.9, 'ncol':1, 'handlelength':1.34, 'labelspacing':0.3}
cbkwargs  = {'orientation':'horizontal', 'fraction':1.3, 'aspect':10}

inclination = 45 # view angle
rot0 = 1 * -90
rot = 1*-40 + rot0 #*1.4 # view angle

prj = ccrs.Orthographic(rot, 90-inclination)
geo = ccrs.Geodetic()
 
#--------------------    

scale = 0.475
fig = plt.figure(figsize=(12.8*scale,4.5*scale))

gs = gridspec.GridSpec(1, 5, width_ratios=[1,0.05,1,1,1])
gs.update(top=0.85, bottom=0.275, left=0.000, right=1-0.005, wspace=0.15)


ax_ODF = fig.add_subplot(gs[0, 0], projection=prj)
ii=1
ax_Y20 = fig.add_subplot(gs[0, ii+1], projection=prj)
ax_Y21 = fig.add_subplot(gs[0, ii+2], projection=prj)
ax_Y22 = fig.add_subplot(gs[0, ii+3], projection=prj)


ax_list = [ax_Y20, ax_Y21, ax_Y22]

#--------------------

lvls = np.linspace(0.0,0.25,6)
nlm = 1/np.sqrt(4*np.pi)*np.array([1,0.5,-0.325, 0,0,0])
plot_ODF(nlm, np.array([[0,2,2],[0,0,-2]]),  ax=ax_ODF, cblbl=r'$\psi/N$', lvls=lvls, tickintvl=2)
ax_ODF.set_title('ODF', pad=8, fontsize=FS)

mrk='k.'
ms=8
mc='#a50f15'

ax = ax_ODF
ax.plot([0],[90], mrk, markersize=ms, color=mc, transform=geo)
ax.text(rot0-60, 63, r'$\vu{z}$', color=mc, horizontalalignment='left', transform=geo)

ax.plot([rot0],[0], mrk, markersize=ms, color=mc, transform=geo)
ax.text(rot0 - 11, 19, r'$\vu{y}$', color=mc, horizontalalignment='left', transform=geo)

ax.plot([rot0-90],[0], mrk, markersize=ms, color=mc, transform=geo)
ax.text(rot0-90 -8, 8, r'$\vu{x}$', color=mc, horizontalalignment='left', transform=geo)


#--------------------

(F,lon,lat) = discretize_fabric([1], np.array([[2,],[0,]]))
plot_Ylm(F,lon,lat, ax=ax_Y20, tlbl='$\hat{\psi}_2^{0}=1$', lbl='$Y_{2}^{0}$')

phi = np.deg2rad(0)
n21 = np.exp(1j*phi)
(F,lon,lat) = discretize_fabric([-np.conj(n21),n21], np.array([[2,2],[1,-1]]))
print('max(imag(F(n21)))=',np.amax(np.imag(F)))
plot_Ylm(F,lon,lat, ax=ax_Y21, tlbl='$\hat{\psi}_2^{1}=1$', lbl=r'$-\big(\hat{\psi}_2^{1}\big)^*Y_{2}^{-1} + \hat{\psi}_2^{1}Y_{2}^{1}$')

phi = np.deg2rad(0)
n22 = np.exp(2j*phi)
(F,lon,lat) = discretize_fabric([np.conj(n22),n22], np.array([[2,2],[2,-2]]))
print('max(imag(F(n22)))=',np.amax(np.imag(F)))
plot_Ylm(F,lon,lat, ax=ax_Y22, tlbl='$\hat{\psi}_2^{2}=1$', lbl=r'$\big(\hat{\psi}_2^{2}\big)^*Y_{2}^{-2} + \hat{\psi}_2^{2}Y_{2}^{2}$')


for ax in ax_list:
    ax.set_global()

#--------------------

fout = 'seriesexpansion.pdf'
print('Saving %s'%(fout))
plt.savefig(fout,dpi=200)
os.system('pdfcrop seriesexpansion.pdf seriesexpansion.pdf')

