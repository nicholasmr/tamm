#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2021

import numpy as np
import copy, sys, code # code.interact(local=locals())
sys.path.insert(0, '../../lib')

from layer import *
from layerstack import *
from synthfabric import *
from fabrictools import *
   
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams, rc
import matplotlib.ticker as mticker
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec

import cartopy.crs as ccrs
import seaborn as sns

FS = 12
rc('font',**{'family':'serif','sans-serif':['Times'],'size':FS})
rc('text', usetex=True)
rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{physics} \usepackage{txfonts} \usepackage{siunitx}'

#---------------------
# Flags
#---------------------

MAKE_SYNTH_FABRICS = 1 # if 1 => remake fabric profiles, if 0 => load stored fabric profiles

#---------------------
# Config
#---------------------

N     = 100 # number of layers
alpha = 0 # angle of incidence (0 = normal incidence)
beta  = np.linspace(1e-3, np.pi +1e-3, N)
E0    = 1.0e3 # Transmitted wave magnitude from layer n=0

#---------------------

fabprofiles = ['ridge', 'icestream', 'shearmargin']

#fabprofiles = ['dome']
#fabprofiles = ['ridge']
#fabprofiles = ['icestream']
#fabprofiles = ['shearmargin']

#---------------------
# Run model
#---------------------

synfab = SynthFabric(N)
if MAKE_SYNTH_FABRICS: synfab.generate_all() # generate synthetic fabric profiles?

for fabprofile in fabprofiles:

    # Load fabric    
    (z, nlm_list, lm_full, N) = synfab.load_profile(fabprofile) # load the save profile
    z = np.linspace(0,-2000, N) # stretch profile slightly so that we cover the entire depth range (doesn't matter, these are synthetic profiles anyway)
    z = np.hstack(([1],z)) # 1m thick surface (isotropic) layer 
    zkm = z*1e-3
    
    # Normalize the ODF coefs for the the transfer-matrix model (only n_2^m needed)
    I = np.array([1, 2, 3, 4, 5])   
    n2m = np.array([ nlm_list[nn,I]/nlm_list[nn,0] for nn in np.arange(N) ]) # normalized l=2 spectral coefs
    lm  = np.array([(2,-2),(2,-1),(2,0),(2,1),(2,2)]).T

    # Setup layer stack and calculate return powers etc.
    lstack = LayerStack(n2m, z, alpha, beta)
    Pm_HH,Pm_HV, dP_HH,dP_HV, c_HHVV, E_HH,E_HV = lstack.get_returns(E0)
   
    #--------------------
    # Plot
    #--------------------
        
    trnpol_colors       = ["#1f78b4", "#e31a1c", "#33a02c", "#6a3d9a", "#33a02c", "#6a3d9a" , "#33a02c", "#6a3d9a"]
    trnpol_infr_colors  = ["#a6cee3", "#fb9a99", "#b2df8a", "#cab2d6", "#b2df8a", "#cab2d6" , "#b2df8a", "#cab2d6"]
    trnpol_trnstr = ["H", "V", "H'", "V'", "H''", "V''", "H'''", "V'''"]
    lw = 1.6
    legkwargs = {'frameon':True, 'fancybox':False, 'edgecolor':'k', 'framealpha':0.9, 'ncol':1, 'handlelength':1.34, 'labelspacing':0.3}
    cbkwargs  = {'orientation':'horizontal', 'fraction':1.3, 'aspect':10}
    
    inclination = 45 # view angle
    #inclination = 0 # view angle
    rot0 = 1 * -90
    rot = 1*-15 + rot0 #*1.4 # view angle
    
    prj = ccrs.Orthographic(rot, 90-inclination)
    geo = ccrs.Geodetic()
 
    #--------------------
    
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

    #--------------------    

    scale = 0.35
    fig = plt.figure(figsize=(25*scale,12*scale))
    gs = fig.add_gridspec(3,7, width_ratios=[0.6,0.15, 0.8, 0.8, 1,1,1], height_ratios=[1,1,0.1])
    al = 0.015
    ar = 0.015
    gs.update(left=al, right=1-ar, top=0.92, bottom=0.10, wspace=0.21, hspace=0.95)
    
    ii = 0
    axODF1      = fig.add_subplot(gs[0+ii, 0], projection=prj)
    axODF2      = fig.add_subplot(gs[1+ii, 0], projection=prj)
    axODF1.set_global(); axODF2.set_global();
    
    ax_eig    = fig.add_subplot(gs[0:2, 2])
    ax_Pm     = fig.add_subplot(gs[0:2, 3], sharey=ax_eig)
    ax_dP_HH  = fig.add_subplot(gs[0:2, 4], sharey=ax_eig)
    ax_dP_HV  = fig.add_subplot(gs[0:2, 5], sharey=ax_eig)
    ax_c_HHVV = fig.add_subplot(gs[0:2, 6], sharey=ax_eig)
    
    ax_dP_HH_cb  = fig.add_subplot(gs[2, 4])
    ax_dP_HV_cb  = fig.add_subplot(gs[2, 5])
    ax_c_HHVV_cb = fig.add_subplot(gs[2, 6])

    #--------------------

    iiODF1, iiODF2 = int(N/2)+1, -1
    fac = 1 # 1/np.sqrt(4*np.pi)
    
    plot_ODF(nlm_list[iiODF1,:], lm_full, ax=axODF1, cblbl=r'$\psi(\SI{%i}{\kilo\metre})$'%(z[iiODF1]*1e-3))
    plot_ODF(nlm_list[iiODF2,:], lm_full, ax=axODF2, cblbl=r'$\psi(\SI{%i}{\kilo\metre})$'%(z[iiODF2]*1e-3))
    
    axODF1.plot([0],[90],'k.', transform=geo)
    axODF1.plot([rot0],[0],'k.', transform=geo)
    axODF1.text(rot0-80, 65, r'$\vu{z}$', horizontalalignment='left', transform=geo)
    axODF1.text(rot0-20, -8, r'$\vu{y}$', horizontalalignment='left', transform=geo)
    
    odfbbox_a = (-0.28,1.43)
    
    subplotkwargs = {'frameon':0, 'alpha':1.0, 'fontsize':FS, 'pad':0.0, }
    
    writeSubplotLabel(axODF1,2,'(a)', bbox=odfbbox_a, **subplotkwargs)
    writeSubplotLabel(axODF2,2,'(b)', bbox=odfbbox_a, **subplotkwargs)

    #--------------------
    
    eigvals, _,_,_     = lstack.get_eigenframe()
    
    ax_eig.plot(eigvals[:,0],zkm[1:],'-k',  lw=lw, label=r'$\lambda_1$')
    ax_eig.plot(eigvals[:,1],zkm[1:],'--k', lw=lw, label=r'$\lambda_2$')
    ax_eig.plot(eigvals[:,2],zkm[1:],':k',  lw=lw, label=r'$\lambda_3$')

    hleg = ax_eig.legend(loc=1,  bbox_to_anchor=(1.2,1), **legkwargs)
    hleg.get_frame().set_linewidth(0.7);
    
    da, daminor = 1, 0.1
    ax_eig.set_yticks([0,-1,-2])
    ax_eig.set_yticks(np.flipud(-np.arange(zkm[0],5,daminor)), minor=True)
    ax_eig.set_ylabel(r'z ($\SI{}{\kilo\metre}$)')
   
    setupAxis(ax_eig, (0.4,0.1), (0,0.8), r'$\lambda_i$', '(c)', spframe=0, frcxlims=1)

    #--------------------

    ax_Pm.plot(Pm_HH, zkm[1:-1],'-k',  lw=lw, label=r'$\overline{P}_{\mathrm{HH}}$')
    ax_Pm.plot(Pm_HV, zkm[1:-1],'--k', lw=lw, label=r'$\overline{P}_{\mathrm{HV}}$')
    
    hleg = ax_Pm.legend(loc=1,  bbox_to_anchor=(1.2,1), **legkwargs)
    hleg.get_frame().set_linewidth(0.7);
    
    
    setupAxis(ax_Pm, (20, 10), (-90,10), r'$\overline{P}$ (dB)', '(d)', spframe=0)
    plt.setp(ax_Pm.get_yticklabels(), visible=False)    
        
    #--------------------
   
    vmin=-10; vmax=-vmin
    ticks = [-10,-5,0,5,10]
    h_HH = plotRxMaps(ax_dP_HH, dP_HH, vmin=vmin,vmax=vmax)
    h_HV = plotRxMaps(ax_dP_HV, dP_HV, vmin=vmin,vmax=vmax)

    plt.setp(ax_dP_HH.get_yticklabels(), visible=False)
    plt.setp(ax_dP_HV.get_yticklabels(), visible=False)
    writeSubplotLabel(ax_dP_HH,2,'(e) $\delta P_{\mathrm{HH}}$',frameon=1, alpha=1.0, fontsize=FS, pad=0.0)
    writeSubplotLabel(ax_dP_HV,2,'(f) $\delta P_{\mathrm{HV}}$',frameon=1, alpha=1.0, fontsize=FS, pad=0.0)
    
    hcb = fig.colorbar(h_HH, ax=ax_dP_HH_cb, extend='both', ticks=ticks[::2], **cbkwargs) 
    hcb.ax.xaxis.set_ticks(ticks, minor=True)
    hcb.ax.set_xlabel(r'$\delta P_{\mathrm{HH}}$ (dB)', labelpad=0)
    ax_dP_HV_cb.set_axis_off()
    
    hcb = fig.colorbar(h_HV, ax=ax_dP_HV_cb, extend='both', ticks=ticks[::2], **cbkwargs) 
    hcb.ax.xaxis.set_ticks(ticks, minor=True)
    hcb.ax.set_xlabel(r'$\delta P_{\mathrm{HV}}$ (dB)', labelpad=0)
    ax_dP_HH_cb.set_axis_off()

   
    #--------------------

    if 1:
        vmin=-180; vmax=-vmin
        ticks = [-180,-90,0,90,180]
        h = plotRxMaps(ax_c_HHVV, np.angle(c_HHVV, deg=True), vmin=vmin,vmax=vmax, cmap='twilight_shifted')
    else:
        vmin=0; vmax=1
        ticks = [0,0.25,0.5,0.75,1]
        h = plotRxMaps(ax_c_HHVV, np.abs(c_HHVV), vmin=vmin,vmax=vmax, cmap='Spectral')

    plt.setp(ax_c_HHVV.get_yticklabels(), visible=False)
    writeSubplotLabel(ax_c_HHVV,2,r'(g) $\varphi_{\mathrm{HV}}$',frameon=1, alpha=1.0, fontsize=FS, pad=0.0)
    
    hcb = fig.colorbar(h, ax=ax_c_HHVV_cb, extend='neither', ticks=ticks[::2], **cbkwargs) # , ticks=ticks
    hcb.ax.xaxis.set_ticks(ticks, minor=True)
    
    hcb.ax.set_xlabel(r'$\varphi_{\mathrm{HV}}$ (\SI{}{\degree})', labelpad=0)
    ax_c_HHVV_cb.set_axis_off()

    #--------------------
    # Save plot
    #--------------------
    
    fout = '%s_alpha%i.png'%(fabprofile, np.rad2deg(alpha))
    print('Saving %s'%(fout))
    plt.savefig(fout,dpi=300)
     