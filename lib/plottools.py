#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2021

import copy, sys, code # code.interact(local=locals())
import numpy as np
import scipy.special as sp

import matplotlib.pyplot as plt
from matplotlib import rcParams, rc
import matplotlib.ticker as mticker
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cartopy.crs as ccrs

FS = 12
rc('font',**{'family':'serif','sans-serif':['Times'],'size':FS})
rc('text', usetex=True)
rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{physics} \usepackage{txfonts} \usepackage{siunitx}'

def discretize_ODF(nlm, lm, latres=60):

    #latres = 60 # latitude resolution on S^2        
    theta = np.linspace(0,   np.pi,   latres) # CO-LAT 
    phi   = np.linspace(0, 2*np.pi, 2*latres) # LON
    phi, theta = np.meshgrid(phi, theta) # gridded 
    lon, colat = phi, theta
    lat = np.pi/2-colat
    
    _,nlm_len = lm.shape
    F = np.real(np.sum([ nlm[ii]*sp.sph_harm(lm[1][ii], lm[0][ii], phi,theta) for ii in np.arange(nlm_len) ], axis=0))
    
    return (F, lon,lat)

def plot_ODF(nlm, lm, ax=None, cmap='Greys', cblabel='', lvls = np.linspace(0.0,0.5,6), tickintvl=4):
    
    pltshow = (ax is None)
    
    if ax is None:
        size = 1.5
        plt.figure(figsize=(size,size))
        inclination = 45 # view angle
        rot0 = -90
        rot = -55 + rot0 # view angle
        prj = ccrs.Orthographic(rot, 90-inclination)
        geo = ccrs.Geodetic()
        ax = plt.subplot(projection=prj)
        ax.set_global()
    
    F, lon,lat = discretize_ODF(nlm, lm)
    hdistr = ax.contourf(np.rad2deg(lon), np.rad2deg(lat), F, transform=ccrs.PlateCarree(), levels=lvls, extend='max', cmap=cmap)

    kwargs_gridlines = {'ylocs':np.arange(-90,90+30,30), 'xlocs':np.arange(0,360+45,45), 'linewidth':0.5, 'color':'black', 'alpha':0.25, 'linestyle':'-'}
    gl = ax.gridlines(crs=ccrs.PlateCarree(), **kwargs_gridlines)
    gl.xlocator = mticker.FixedLocator(np.array([-135, -90, -45, 0, 90, 45, 135, 180]))

    cb1 = plt.colorbar(hdistr, ax=ax, fraction=0.075, aspect=9,  orientation='horizontal', pad=0.1, ticks=lvls[::tickintvl])   
    cb1.set_label(cblabel)
    cb1.ax.xaxis.set_ticks(lvls, minor=True)
    
    if pltshow: plt.show()
    
    return hdistr

def plot_returns(z, returns):
    
    Pm_HH,Pm_HV, dP_HH,dP_HV, c_HHVV, E_HH,E_HV = returns
    zkm = 1e-3 * z
    
    def writeSubplotLabel(ax,loc,txt,frameon=True, alpha=1.0, fontsize=FS, pad=0.15, ma='none', bbox=None):
        at = AnchoredText(txt, loc=loc, prop=dict(size=fontsize), frameon=frameon, bbox_to_anchor=bbox, bbox_transform=ax.transAxes)
        at.patch.set_linewidth(0.7)
        ax.add_artist(at)
    
    def plotRxMaps(ax, Rx, vmin=None, vmax=None, cmap='RdBu_r'):
        
        extent = [0,180,zkm[-1],0]
        h = ax.imshow(Rx, vmin=vmin, vmax=vmax, cmap=cmap, extent=extent)
        ax.set_xlabel(r'$\beta$ ($\SI{}{\degree}$)')
        ax.set_aspect(aspect="auto")
        ticks = [0,45,90,90+45,180]
        ax.set_xticks(ticks[::2])
        ax.set_xticks(ticks, minor=True)
        return h
    
    def setupAxis(ax, daam, xminmax, xlbl, splbl, spframe=1, frcxlims=0):

        xlim = ax.get_xlim()
        da, daminor = daam
        xmin,xmax = xminmax
        ax.set_xticks(np.arange(xmin,xmax+da,da))
        ax.set_xticks(np.arange(xmin,xmax+da,daminor), minor=True)
        ax.set_xlim(xlim if not frcxlims else xminmax)
        ax.set_xlabel(xlbl)
        writeSubplotLabel(ax, 2, splbl, frameon=spframe, alpha=1.0, fontsize=FS, pad=0.0)

    def setcb(ax, h, ticks=[], xlbl='', phantom=False):
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="3.5%", pad=0.6)
        if not phantom:
            hcb = plt.colorbar(h, cax=cax, extend='both', ticks=ticks[::2], **cbkwargs) 
            hcb.ax.xaxis.set_ticks(ticks, minor=True)
            hcb.ax.set_xlabel(xlbl, labelpad=0)
        else:
            cax.set_axis_off()

    #--------------------    

    scale = 0.3
    fig = plt.figure(figsize=(20*scale,15*scale))

    gs = gridspec.GridSpec(1, 4)
    gs.update(top=0.965, bottom=0.11, left=-0.02, right=1-0.02, wspace=0.17, hspace=0.95)
    ax_Pm     = fig.add_subplot(gs[0, 0])
    ax_dP_HH  = fig.add_subplot(gs[0, 1], sharey=ax_Pm)
    ax_dP_HV  = fig.add_subplot(gs[0, 2], sharey=ax_Pm)
    ax_c_HHVV = fig.add_subplot(gs[0, 3], sharey=ax_Pm)
    
    lw = 1.6
    legkwargs = {'frameon':True, 'fancybox':False, 'edgecolor':'k', 'framealpha':0.9, 'ncol':1, 'handlelength':1.34, 'labelspacing':0.3}
    cbkwargs  = {'orientation':'horizontal', 'fraction':1.3, 'aspect':10}
    
    #--------------------

    ax_Pm.plot(Pm_HH, zkm[1:-1],'-k',  lw=lw, label=r'$\overline{P}_{\mathrm{HH}}$')
    ax_Pm.plot(Pm_HV, zkm[1:-1],'--k', lw=lw, label=r'$\overline{P}_{\mathrm{HV}}$')
    hleg = ax_Pm.legend(loc=2,  **legkwargs)
    hleg.get_frame().set_linewidth(0.7);        
    setupAxis(ax_Pm, (20, 10), (-150,10), r'$\overline{P}$ (dB)', '', spframe=0)
    #
    da, daminor = -0.5, -0.1
    ax_Pm.set_yticks(np.arange(zkm[1],zkm[-1]*1.1,da))
    ax_Pm.set_yticks(np.arange(zkm[1],zkm[-1]*1.1,daminor), minor=True)
    ax_Pm.set_ylabel(r'z ($\SI{}{\kilo\metre}$)')
    #
    setcb(ax_Pm, 0, phantom=True)
    
    #--------------------
   
    vmin=-10; vmax=-vmin
    ticks = [-10,-5,0,5,10]
    h_HH = plotRxMaps(ax_dP_HH, dP_HH, vmin=vmin,vmax=vmax)
    h_HV = plotRxMaps(ax_dP_HV, dP_HV, vmin=vmin,vmax=vmax)
    plt.setp(ax_dP_HH.get_yticklabels(), visible=False)
    plt.setp(ax_dP_HV.get_yticklabels(), visible=False)
    writeSubplotLabel(ax_dP_HH,2,'$\delta P_{\mathrm{HH}}$',frameon=1, alpha=1.0, fontsize=FS, pad=0.0)
    writeSubplotLabel(ax_dP_HV,2,'$\delta P_{\mathrm{HV}}$',frameon=1, alpha=1.0, fontsize=FS, pad=0.0)
    setcb(ax_dP_HH, h_HH, ticks=ticks, xlbl=r'$\delta P_{\mathrm{HH}}$ (dB)')
    setcb(ax_dP_HV, h_HV, ticks=ticks, xlbl=r'$\delta P_{\mathrm{HV}}$ (dB)')
    
    #--------------------

    vmin=-180; vmax=-vmin
    ticks = [-180,-90,0,90,180]
    h = plotRxMaps(ax_c_HHVV, np.angle(c_HHVV, deg=True), vmin=vmin,vmax=vmax, cmap='twilight_shifted')
    plt.setp(ax_c_HHVV.get_yticklabels(), visible=False)
    writeSubplotLabel(ax_c_HHVV,2,r'$\varphi_{\mathrm{HV}}$',frameon=1, alpha=1.0, fontsize=FS, pad=0.0)
    setcb(ax_c_HHVV, h, ticks=ticks, xlbl=r'$\varphi_{\mathrm{HV}}$ (\SI{}{\degree})')

    #--------------------

    plt.show()

