#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2021

import copy, sys, code # code.interact(local=locals())
import numpy as np
import scipy.special as sp

from layer import *

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

# Default ODF plotting params
lvls_default      = np.linspace(0.0,0.5,6)
tickintvl_default = 5


def plot_returns(z, returns, a2, eigvals, vminmax=15, I0=1, nlm_true=None, lm_true=None, lvls=lvls_default, tickintvl=tickintvl_default):

    """
        Inputs:
        -------
                
        z:          Distance from surface
        returns:    (Pm_HH,Pm_HV, dP_HH,dP_HV, c_HHVV, E_HH,E_HV)
        a2:         Second-order structure tensor
        eigvals:    Eigenvalues
        vminmax:    Colorbar min/max range for delta(P_ij) plots
        I0:         First layer to plot (0=sfc)
        nlm_true:   For plotting, use these ODF harmonic coefs instead of determining them from a2 (which contains only n_2^m modes).
        lm_true:    lm list for nlm_true
        lvls:       Contour levels for plotting ODFs
        tickintvl:  Colorbar tick label interval for plotting ODFs
    """    
    
    Pm_HH,Pm_HV, dP_HH,dP_HV, c_HHVV, E_HH,E_HV = returns

    zkm = 1e-3 * z
    frameon = 1 # Frame on subplot labels?
    #I0 = 1 # Layer index of first layer to plot (if I0=1 then skip surface reflection in results)
    
    #--------------------    
    
    inclination = 45 # view angle
    rot0 = -90
    rot = -55 + rot0 # view angle
    
    prj = ccrs.Orthographic(rot, 90-inclination)
    geo = ccrs.Geodetic()

    #--------------------

    scale = 0.35
    fig = plt.figure(figsize=(25*scale,13*scale))
    
    gs_master = gridspec.GridSpec(1, 2, width_ratios=[1/6, 5/6])
    gs_master.update(top=0.965, bottom=0.11, left=-0.0175, right=1-0.02, wspace=0.065)
    
    gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_master[0], hspace=0.70)
    ax_ODFs = [fig.add_subplot(gs[ii, 0], projection=prj) for ii in np.arange(3)]
       
    k = 0.9
    gs = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs_master[1], wspace=0.17, hspace=0.95, width_ratios=[k,k,1,1,1])
    ax_eig    = fig.add_subplot(gs[0, 0])
    ax_Pm     = fig.add_subplot(gs[0, 1], sharey=ax_eig)
    ax_dP_HH  = fig.add_subplot(gs[0, 2], sharey=ax_eig)
    ax_dP_HV  = fig.add_subplot(gs[0, 3], sharey=ax_eig)
    ax_c_HHVV = fig.add_subplot(gs[0, 4], sharey=ax_eig)
    
    lw = 1.6
    legkwargs = {'frameon':True, 'fancybox':False, 'edgecolor':'k', 'framealpha':0.85, 'ncol':1, 'handlelength':1.34, 'labelspacing':0.3}
    
    #--------------------
    
    IplotODF = [1, int(len(z)/2)-1, -1] # Plot ODF at these 3 time steps
    if nlm_true is None:
        nlm = np.zeros((3,6), dtype=np.complex128)
        for ii in np.arange(3): nlm[ii,:], lm = c2_to_nlm(a2[IplotODF[ii],:,:]) 
    else:
        nlm = np.zeros((3,len(nlm_true[0,:])), dtype=np.complex128)
        for ii in np.arange(3): nlm[ii,:], lm = nlm_true[IplotODF[ii],:], lm_true
    plot_ODFs(ax_ODFs, nlm, lm, zkm[IplotODF], geo, rot0, lvls=lvls, tickintvl=tickintvl)
        
    #--------------------
    
    dzhalf = -np.abs(zkm[0]-zkm[1])/2
    ax_eig.plot(eigvals[I0:,0], zkm[(I0):-1]+dzhalf, '-k',  lw=lw, label=r'$\lambda_1$')
    ax_eig.plot(eigvals[I0:,1], zkm[(I0):-1]+dzhalf, '--k', lw=lw, label=r'$\lambda_2$')
    ax_eig.plot(eigvals[I0:,2], zkm[(I0):-1]+dzhalf, ':k',  lw=lw, label=r'$\lambda_3$')
    
    hleg = ax_eig.legend(loc=1, bbox_to_anchor=(1.125,1), **legkwargs)
    hleg.get_frame().set_linewidth(0.7);
    
    da, daminor = 1, 0.1
    ax_eig.set_yticks([0,-1,-2])
    ax_eig.set_yticks(np.arange(zkm[-1],zkm[I0],daminor), minor=True)
    ax_eig.set_ylim([zkm[-1],zkm[I0]])
    ax_eig.set_ylabel(r'z ($\SI{}{\kilo\metre}$)', fontsize=FS)
       
    setupAxis(ax_eig, (0.5,0.1), (0,1.0), r'$\lambda_n$', r'{\bf d}', spframe=frameon, frcxlims=1)
    setcb(ax_eig, 0, phantom=True)
    
    #--------------------

    ax_Pm.plot(Pm_HH[I0:], zkm[(I0+1):-1], '-k',  lw=lw, label=r'$\overline{P}_{\mathrm{HH}}$')
    ax_Pm.plot(Pm_HV[I0:], zkm[(I0+1):-1], '--k', lw=lw, label=r'$\overline{P}_{\mathrm{HV}}$')
    
    hleg = ax_Pm.legend(loc=1, bbox_to_anchor=(1.125,1), **legkwargs)
    hleg.get_frame().set_linewidth(0.7);        
    setupAxis(ax_Pm, (20, 10), (-200,10), r'$\overline{P}_{jk}$ (dB)', r'{\bf e}', spframe=frameon)
    plt.setp(ax_Pm.get_yticklabels(), visible=False)   
    setcb(ax_Pm, 0, phantom=True)
    
    #--------------------
   
    vmin=-vminmax; vmax=-vmin
    ticks = np.arange(vmin, vmax+1e-5, vmax/2)
    h_HH = plotRxMaps(ax_dP_HH, dP_HH, zkm, I0, vmin=vmin,vmax=vmax)
    h_HV = plotRxMaps(ax_dP_HV, dP_HV, zkm, I0, vmin=vmin,vmax=vmax)
    plt.setp(ax_dP_HH.get_yticklabels(), visible=False)
    plt.setp(ax_dP_HV.get_yticklabels(), visible=False)
    writeSubplotLabel(ax_dP_HH, 2, r'{\bf f}', frameon=frameon, alpha=1.0, fontsize=FS)
    writeSubplotLabel(ax_dP_HV, 2, r'{\bf g}', frameon=frameon, alpha=1.0, fontsize=FS)
    setcb(ax_dP_HH, h_HH, ticks=ticks, xlbl=r'$\delta P_{\mathrm{HH}}$ (dB)')
    setcb(ax_dP_HV, h_HV, ticks=ticks, xlbl=r'$\delta P_{\mathrm{HV}}$ (dB)')
    
    #--------------------

    vmin=-180; vmax=-vmin
    ticks = [-180,-90,0,90,180]
    h = plotRxMaps(ax_c_HHVV, np.angle(c_HHVV, deg=True), zkm, I0, vmin=vmin,vmax=vmax, cmap='twilight_shifted')
    plt.setp(ax_c_HHVV.get_yticklabels(), visible=False)
    writeSubplotLabel(ax_c_HHVV, 2, r'{\bf h}' ,frameon=frameon, alpha=1.0, fontsize=FS)
    setcb(ax_c_HHVV, h, ticks=ticks, xlbl=r'$\varphi_{\mathrm{HV}}$ (\SI{}{\degree})')

    #--------------------

    return (plt, ax_ODFs, ax_eig, ax_Pm, ax_dP_HH, ax_dP_HV, ax_c_HHVV, IplotODF, prj,geo,rot0)


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


def plot_ODF(nlm, lm, ax=None, cmap='Greys', cblabel='$\psi$', lvls=lvls_default, tickintvl=tickintvl_default):
    
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
    hdistr = ax.contourf(np.rad2deg(lon), np.rad2deg(lat), F, transform=ccrs.PlateCarree(), levels=lvls, extend=('max' if lvls[0]==0.0 else 'both'), cmap=cmap)

    kwargs_gridlines = {'ylocs':np.arange(-90,90+30,30), 'xlocs':np.arange(0,360+45,45), 'linewidth':0.5, 'color':'black', 'alpha':0.25, 'linestyle':'-'}
    gl = ax.gridlines(crs=ccrs.PlateCarree(), **kwargs_gridlines)
    gl.xlocator = mticker.FixedLocator(np.array([-135, -90, -45, 0, 90, 45, 135, 180]))

    cb1 = plt.colorbar(hdistr, ax=ax, fraction=0.075, aspect=9,  orientation='horizontal', pad=0.1, ticks=lvls[::tickintvl])   
    cb1.set_label(cblabel)
    cb1.ax.xaxis.set_ticks(lvls, minor=True)
    
    if pltshow: plt.show()
    
    return hdistr


def plot_ODFs(ax_ODFs, nlm, lm, zkm, geo, rot0, ODFsymb=r'\psi', panelno='a', lvls=lvls_default, tickintvl=tickintvl_default):
    
    for ii in np.arange(len(ax_ODFs)):
        
        ax = ax_ODFs[ii]
        plot_ODF(nlm[ii,:], lm, ax=ax, cblabel=r'$%s(\SI{%1.0f}{\kilo\metre})/N$'%(ODFsymb, zkm[ii]), lvls=lvls, tickintvl=tickintvl)
        
        if ii == 0:
            colorxi = '#a50f15'
            ax.plot([0],[90],'.',       color=colorxi, transform=geo)
            ax.plot([rot0],[0],'.',     color=colorxi, transform=geo)
            ax.plot([2*rot0],[0],'.',   color=colorxi, transform=geo)
            ax.text(rot0-120, 62,   r'$\vu{z}$', color=colorxi, horizontalalignment='left', transform=geo)
            ax.text(rot0-27, -0,    r'$\vu{y}$', color=colorxi, horizontalalignment='left', transform=geo)
            ax.text(2*rot0+10, -8,  r'$\vu{x}$', color=colorxi, horizontalalignment='left', transform=geo)
        
        subplotkwargs = {'frameon':1, 'alpha':1.0, 'fontsize':FS, 'pad':0.275, }
        writeSubplotLabel(ax,2,r'{\bf %s}'%(chr(ord(panelno)+ii)), bbox=(-0.35,1.25), **subplotkwargs)
        
        ax.set_global()


def writeSubplotLabel(ax, loc, txt, frameon=True, alpha=1.0, fontsize=FS, pad=0.3, ma='none', bbox=None):
    at = AnchoredText(txt, loc=loc, prop=dict(size=fontsize), frameon=frameon, pad=pad, bbox_to_anchor=bbox, bbox_transform=ax.transAxes)
    at.patch.set_linewidth(0.7)
    ax.add_artist(at)


def plotRxMaps(ax, Rx, zkm, I0, vmin=None, vmax=None, cmap='RdBu_r'):
    
    extent = [0,180,zkm[-2],zkm[I0]]
    h = ax.imshow(Rx[I0:,:], vmin=vmin, vmax=vmax, cmap=cmap, extent=extent)
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
    writeSubplotLabel(ax, 2, splbl, frameon=spframe, alpha=1.0, fontsize=FS)


def setcb(ax, h, ticks=[], xlbl='', phantom=False):
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="3.5%", pad=0.6)
    if not phantom:
        cbkwargs  = {'orientation':'horizontal', 'fraction':1.3, 'aspect':10}
        hcb = plt.colorbar(h, cax=cax, extend='both', ticks=ticks[::2], **cbkwargs) 
        hcb.ax.xaxis.set_ticks(ticks, minor=True)
        hcb.ax.set_xlabel(xlbl, labelpad=0)
    else:
        cax.set_axis_off()

