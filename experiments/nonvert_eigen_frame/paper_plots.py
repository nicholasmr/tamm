#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2021

import numpy as np
import copy, sys, os, code # code.interact(local=locals())
sys.path.insert(0, '../../lib')

from specfabpy import specfabpy as sf

from layer import *
from layerstack import *
from plottools import *

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams, rc
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cartopy.crs as ccrs

#---------------------
# Config
#---------------------

### Layer stack config

DEBUG = 0

RESMUL = 1 if DEBUG else 6
N_layers = 50*RESMUL + 1 # Number of layers
N_frames = 20*RESMUL # Number of rotated frames of the radar system 

H = 2000 # Ice column height
z = np.linspace(0,-H, N_layers) # Interface positions

FABRIC_TYPE = 2  # 1 = single max, 2 = girdle

### Radar config

alpha  = np.deg2rad(0)  # Angle of incidence for reference experiments
alphaX = np.deg2rad(10) # Angle of incidence of non-normally incident waves

E0    = 1.0e3   # Transmitted wave magnitude from layer n=0
freq  = 179.0e6 # Transmitted frequency from layer n=0

### Transfer matrix model config

#mt = 'FP' # Fujita--Paren
mt = 'GTM' # GTM (Rathmann et al., 2021)

### Other

DO_DIAGNOSTIC = 1 # Plot truncated profile results?

#---------------------
# Setup
#---------------------

### Specfab

nlm_len = sf.init(4)
nlm_list = np.zeros((N_layers,nlm_len), dtype=np.complex128)
nlm_list[:,0] = 1/np.sqrt(4*np.pi) # normalized distribution

### Construct idealized fabric profiles

# Single max.
if FABRIC_TYPE == 1: 
    n20 = 0.2 * 3.0 
    n21 = 0
    n22 = 0
    
# Girdle
if FABRIC_TYPE == 2: 
    n20 = 0.15 
    n21 = 0
    n22 = -3/2 * np.sqrt(2/3) * n20

# Strengthen fabric profiles with depth
linvec = np.linspace(0.5,1, N_layers-1)
nlm_list[1:,3] =  n20*linvec # n20
nlm_list[1:,4] = +n21*linvec # n21p
nlm_list[1:,5] = +n22*linvec # n22p
nlm_list[1:,2] = -np.conj(nlm_list[1:,4]) # n21m
nlm_list[1:,1] = +np.conj(nlm_list[1:,5]) # n22m
    
if DEBUG:
    print('<c^2> in first and last layer are:')
    print(nlm_to_c2(nlm_list[0,:]))
    print(nlm_to_c2(nlm_list[-1,:]))

# Rotate ODF profile toward horizontal plane
angles = np.deg2rad(np.linspace(0,-90, N_layers-1))
for nn,b in enumerate(angles,1):
    c, s = np.cos(b), np.sin(b)
    Qy = np.array(((c, 0, s), (0,1,0), (-s,0,c))) # Rotation matrix
    c2 = nlm_to_c2(nlm_list[nn,:]) 
    c2_rot = np.matmul(np.matmul(Qy, c2), Qy.T)
    nlm_list[nn,:6], lm_list = c2_to_nlm(c2_rot) # Set rotated ODF spectral coefs

#---------------------
# Run model
#---------------------

### Reference profile
    
lstack = LayerStack(nlm_list, z, N_frames=N_frames, modeltype=mt) # init layer stack
returns = lstack.get_returns(E0, f=freq, alpha=alpha) # get returns for radar config
Pm_HH,Pm_HV, dP_HH,dP_HV, c_HHVV, E_HH,E_HV = returns # unpack
eigvals, e1,e2,e3, a2 = lstack.get_eigenframe() 

lm   = lstack.lm   
zkm  = lstack.z*1e-3
dzkm = lstack.dz*1e-3

if DO_DIAGNOSTIC:
    
    # Truncated profile of reference
    nlm_list_trunc = nlm_list.copy()
    nlm_list_trunc[:,np.array([2,4])] = 0
    
    lstack_trunc = LayerStack(nlm_list_trunc, z, N_frames=N_frames, modeltype=mt)
    returns_trunc = lstack_trunc.get_returns(E0, f=freq, alpha=alpha)
    Pm_HH_trunc,Pm_HV_trunc, dP_HH_trunc,dP_HV_trunc, c_HHVV_trunc, E_HH_trunc,E_HV_trunc = returns_trunc
    _, e1_trunc,e2_trunc,e3_trunc, a2_trunc = lstack_trunc.get_eigenframe()

### Non-normally incident model

if DO_DIAGNOSTIC:
    
    lstack_alphaX = LayerStack(nlm_list, z, N_frames=N_frames, modeltype=mt)
    returns_alphaX = lstack_alphaX.get_returns(E0, f=freq, alpha=alphaX)
    _,_, dP_HH_alphaX,_, _, _,_ = returns_alphaX
    
    # Truncated profile of non-normally incident model
    lstack_trunc_alphaX = LayerStack(nlm_list_trunc, z, N_frames=N_frames, modeltype=mt)
    returns_trunc_alphaX = lstack_trunc_alphaX.get_returns(E0, f=freq, alpha=alphaX)
    _,_, dP_HH_trunc_alphaX,_, _, _,_ = returns_trunc_alphaX

#--------------------
# Plot setup
#--------------------

lw = 1.6
legkwargs = {'frameon':True, 'fancybox':False, 'edgecolor':'k', 'framealpha':0.85, 'ncol':1, 'handlelength':1.34, 'labelspacing':0.3}

inclination = 45 # view angle
rot0 = -90
rot = -55 + rot0 # view angle

prj = ccrs.Orthographic(rot, 90-inclination)
geo = ccrs.Geodetic()

frameon = 1 # Frame on subplot labels?
I0 = 1 # Subsurface layer index (if =1 then skip surface reflection in results)

IaxODF = [0,1,2]
IplotODF   = [1, int(N_layers*0.47)+1, -1] # time steps to plot ODF for

### Some common plotting functions

def writeSubplotLabel(ax, loc, txt, frameon=True, alpha=1.0, fontsize=FS, pad=0.3, ma='none', bbox=None):
  
    at = AnchoredText(txt, loc=loc, prop=dict(size=fontsize), frameon=frameon, pad=pad, bbox_to_anchor=bbox, bbox_transform=ax.transAxes)
    at.patch.set_linewidth(0.7)
    ax.add_artist(at)

def plotRxMaps(ax, Rx, vmin=None, vmax=None, cmap='RdBu_r'):
    
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

def plot_ODFs(axODFs, IplotODF, nlm_list, ei_ref, ODFsymb=r'\psi'):
    
    IaxODF = np.arange(len(axODFs))
    
    for ii in IaxODF:
        
        nlm_ii = IplotODF[ii]
        ax     = axODFs[ii]
        
        plot_ODF(nlm_list[nlm_ii,:], lm_list, ax=axODFs[ii], cblabel=r'$%s(\SI{%1.0f}{\kilo\metre})/N$'%(ODFsymb,z[nlm_ii]*1e-3))
        
        if ii == 0:
            
            colorxi = '#a50f15'
            
            ax.plot([0],[90],'.',       color=colorxi, transform=geo)
            ax.plot([rot0],[0],'.',     color=colorxi, transform=geo)
            ax.plot([2*rot0],[0],'.',   color=colorxi, transform=geo)
            
            ax.text(rot0-120, 62,   r'$\vu{z}$', color=colorxi, horizontalalignment='left', transform=geo)
            ax.text(rot0-27, -0,    r'$\vu{y}$', color=colorxi, horizontalalignment='left', transform=geo)
            ax.text(2*rot0+10, -8,  r'$\vu{x}$', color=colorxi, horizontalalignment='left', transform=geo)
        
        #-----
        
        def getAngles(vec):
            
            x,y,z = vec
            phi   = np.rad2deg(np.arctan2(y,x))
            theta = 90 - np.rad2deg(np.arccos(z))
            return (theta,phi)
        
        def plot_ei(vec, mrk='o', lbl=False, phi_rel=0, theta_rel=-18):
            
            theta,phi = getAngles(vec)
            if 0<=phi<=180 and (theta<=0): theta,phi = getAngles(-vec)
            color = '#08519c'
            ax.plot([phi],[theta], mrk, ms=7.0, markerfacecolor="None", markeredgecolor=color, markeredgewidth=1.1, transform=geo)
            
            if lbl: 
                # Plotting ei_ref==e1 (SM expr) or ei_ref==e3 (girdle expr)? Test the z comp of the vector plotted...
                if np.abs(vec[2])> 0.01: ax.text(rot0-phi+phi_rel, theta+theta_rel, r'$\vb{e}_1$', horizontalalignment='left', color=color, transform=geo)
                else:                    ax.text(phi-22,theta+8, r'$\vb{e}_3$', horizontalalignment='left', color=color, transform=geo)

        #-----
                
        plot_ei(ei_ref[nlm_ii,:], lbl=(ii==0)) # Plot the passed "reference" eigen direction
        
        subplotkwargs = {'frameon':1, 'alpha':1.0, 'fontsize':FS, 'pad':0.275, }
        writeSubplotLabel(ax,2,r'{\bf %s}'%(chr(ord('a')+ii)), bbox=(-0.35,1.25), **subplotkwargs)
    
        ax.set_global()

#--------------------
# Plot reference experiment
#--------------------   

if 1:
        
    scale = 0.35
    fig = plt.figure(figsize=(25*scale,13*scale))
    
    gs_master = gridspec.GridSpec(1, 2, width_ratios=[1/6, 5/6])
    gs_master.update(top=0.965, bottom=0.11, left=-0.0175, right=1-0.02, wspace=0.065)
    
    gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_master[0], hspace=0.70)
    axODFs = [fig.add_subplot(gs[ii, 0], projection=prj) for ii in IaxODF]
       
    k = 0.9
    gs = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs_master[1], wspace=0.17, hspace=0.95, width_ratios=[k,k,1,1,1])
    ax_eig    = fig.add_subplot(gs[0, 0])
    ax_Pm     = fig.add_subplot(gs[0, 1], sharey=ax_eig)
    ax_dP_HH  = fig.add_subplot(gs[0, 2], sharey=ax_eig)
    ax_dP_HV  = fig.add_subplot(gs[0, 3], sharey=ax_eig)
    ax_c_HHVV = fig.add_subplot(gs[0, 4], sharey=ax_eig)
    
    #--------------------
    
    plot_ODFs(axODFs, IplotODF, nlm_list, e1 if FABRIC_TYPE==1 else e3)
    
    #--------------------
       
    vmin = -15; vmax = -vmin
    ticks = np.arange(vmin, vmax+1e-5, vmax/2)
    
    h_HH = plotRxMaps(ax_dP_HH, dP_HH, vmin=vmin,vmax=vmax)
    h_HV = plotRxMaps(ax_dP_HV, dP_HV, vmin=vmin,vmax=vmax)
    
    plt.setp(ax_dP_HH.get_yticklabels(), visible=False)
    plt.setp(ax_dP_HV.get_yticklabels(), visible=False)
    writeSubplotLabel(ax_dP_HH,2,r'{\bf f}',frameon=frameon, alpha=1.0, fontsize=FS)
    writeSubplotLabel(ax_dP_HV,2,r'{\bf g}',frameon=frameon, alpha=1.0, fontsize=FS)
    setcb(ax_dP_HH, h_HH, ticks=ticks, xlbl=r'$\delta P_{\mathrm{HH}}$ (dB)')
    setcb(ax_dP_HV, h_HV, ticks=ticks, xlbl=r'$\delta P_{\mathrm{HV}}$ (dB)')
    
    #--------------------
    
    vmin = -180; vmax = -vmin
    ticks = [-180,-90,0,90,180]
    
    h = plotRxMaps(ax_c_HHVV, np.angle(c_HHVV, deg=True), vmin=vmin,vmax=vmax, cmap='twilight_shifted')
    plt.setp(ax_c_HHVV.get_yticklabels(), visible=False)
    writeSubplotLabel(ax_c_HHVV,2,r'{\bf h}',frameon=frameon, alpha=1.0, fontsize=FS)
    setcb(ax_c_HHVV, h, ticks=ticks, xlbl=r'$\varphi_{\mathrm{HV}}$ (\SI{}{\degree})')
    
    #--------------------
    
    dzhalf = -dzkm[I0:]/2
    ax_eig.plot(eigvals[I0:,0], zkm[(I0):-1]+dzhalf, '-k',  lw=lw, label=r'$\lambda_1$')
    ax_eig.plot(eigvals[I0:,1], zkm[(I0):-1]+dzhalf, '--k', lw=lw, label=r'$\lambda_2$')
    ax_eig.plot(eigvals[I0:,2], zkm[(I0):-1]+dzhalf, ':k',  lw=lw, label=r'$\lambda_3$')
    
    if FABRIC_TYPE==1: hleg = ax_eig.legend(loc=8, **legkwargs) 
    if FABRIC_TYPE==2: hleg = ax_eig.legend(loc=1, bbox_to_anchor=(1.125,1), **legkwargs)
    hleg.get_frame().set_linewidth(0.7);
    
    da, daminor = 1, 0.1
    ax_eig.set_yticks([0,-1,-2])
    ax_eig.set_yticks(np.flipud(-np.arange(zkm[I0],3,daminor)), minor=True)
    ax_eig.set_ylim([zkm[-1],zkm[I0]])
    ax_eig.set_ylabel(r'z ($\SI{}{\kilo\metre}$)', fontsize=FS)
       
    setupAxis(ax_eig, (0.5,0.1), (0,1.0), r'$\lambda_n$', r'{\bf d}', spframe=frameon, frcxlims=1)
    setcb(ax_eig, 0, phantom=True)
    
    #--------------------
    
    ax_Pm.plot(Pm_HH[I0:], zkm[(I0+1):-1],'-k',  lw=lw, label=r'$\overline{P}_{\mathrm{HH}}$')
    ax_Pm.plot(Pm_HV[I0:], zkm[(I0+1):-1],'--k', lw=lw, label=r'$\overline{P}_{\mathrm{HV}}$')
    
    if FABRIC_TYPE==1: hleg = ax_Pm.legend(loc=2, bbox_to_anchor=(0,0.875), **legkwargs)
    if FABRIC_TYPE==2: hleg = ax_Pm.legend(loc=2, bbox_to_anchor=(0,0.775), **legkwargs)
    hleg.get_frame().set_linewidth(0.7);
    
    
    setupAxis(ax_Pm, (20, 10), (-110,10), r'$\overline{P}_{jk}$ (dB)', r'{\bf e}', spframe=frameon)
    plt.setp(ax_Pm.get_yticklabels(), visible=False)    
    setcb(ax_Pm, 0, phantom=True)

    #--------------------

    # Save plot
    fout = 'syntheticprofile_fabtype%i.png'%(FABRIC_TYPE)
    print('Saving %s'%(fout))
    plt.savefig(fout, dpi=300)
     
#--------------------
# Plot truncated and non-normally incident experiments
#--------------------   

if DO_DIAGNOSTIC:
    
    scale = 0.35
    fig = plt.figure(figsize=(18*scale,13*scale))
    
    r = 0.28
    gs_master = gridspec.GridSpec(1, 2, width_ratios=[r, 1-r])
    gs_master.update(top=0.97, bottom=0.11, left=-0.04, right=1-0.08, wspace=0.065)
    
    gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_master[0], hspace=0.70)
    axODFs = [fig.add_subplot(gs[ii, 0], projection=prj) for ii in IaxODF]
       
    gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_master[1], wspace=0.17, hspace=0.95, width_ratios=[1,1,1])
    ax_dP_HH        = fig.add_subplot(gs[0, 0])
    ax_dP_HH_alphaX = fig.add_subplot(gs[0, 1], sharey=ax_dP_HH)
    ax_E            = fig.add_subplot(gs[0, 2], sharey=ax_dP_HH)
    
    #--------------------
    
    plot_ODFs(axODFs, IplotODF, nlm_list_trunc, e1_trunc if FABRIC_TYPE==1 else e3_trunc, ODFsymb=r'\psi^\dagger')
    
    #--------------------
    
    vmin = -0.5; vmax = -vmin
    ticks = np.arange(vmin, vmax+1e-5, vmax/2)
    
    Z_alpha0 = dP_HH - dP_HH_trunc
    h_HH = plotRxMaps(ax_dP_HH, Z_alpha0, vmin=vmin,vmax=vmax)
    writeSubplotLabel(ax_dP_HH,2,r'{\bf d}',frameon=frameon, alpha=1.0, fontsize=FS)
    writeSubplotLabel(ax_dP_HH,1,r'$\alpha=\SI{%i}{\degree}$'%(np.rad2deg(alpha)),frameon=frameon, alpha=1.0, fontsize=FS)
    setcb(ax_dP_HH, h_HH, ticks=ticks, xlbl=r'$\delta P_{\mathrm{HH}} - \delta P_{\mathrm{HH}}^\dagger$ (dB)')
    
    ax = ax_dP_HH
    da, daminor = 1, 0.1
    ax.set_yticks([0,-1,-2])
    ax.set_yticks(np.flipud(-np.arange(zkm[I0],5,daminor)), minor=True)
    ax.set_ylabel(r'z ($\SI{}{\kilo\metre}$)', fontsize=FS)
    ax.set_ylim(zkm[[-1,I0]])
    
    #--------------------

    vmin = -15; vmax = -vmin
    ticks = np.arange(vmin, vmax+1e-5, vmax/2)

    Z_alphaX = dP_HH_alphaX - dP_HH_trunc_alphaX
    h_HH = plotRxMaps(ax_dP_HH_alphaX, Z_alphaX, vmin=vmin,vmax=vmax)
    writeSubplotLabel(ax_dP_HH_alphaX,2,r'{\bf e}',frameon=frameon, alpha=1.0, fontsize=FS)
    writeSubplotLabel(ax_dP_HH_alphaX,1,r'$\alpha=\SI{%i}{\degree}$'%(np.rad2deg(alphaX)),frameon=frameon, alpha=1.0, fontsize=FS)
    setcb(ax_dP_HH_alphaX, h_HH, ticks=ticks, xlbl=r'$\delta P_{\mathrm{HH}} - \delta P_{\mathrm{HH}}^\dagger$ (dB)')

    plt.setp(ax_dP_HH_alphaX.get_yticklabels(), visible=False)
    
    #--------------------

    ### Monocrystal rheological params
    
    # Use linear grain rheology from Rathmann and Lilien (2021)
    nprime = 1 
    Ecc    = sf.ecc_opt_lin 
    Eca    = sf.eca_opt_lin
    g      = sf.alpha_opt_lin
    
    ### Calculate enhancement factors
    
    Eii_true  = np.zeros((N_layers,3))
    Eii_trunc = np.zeros((N_layers,3))

    Eij_true  = np.zeros((N_layers,3))
    Eij_trunc = np.zeros((N_layers,3))
    
    for nn in np.arange(N_layers)[:]:
        
        v1,v2,v3, eigvals = sf.frame(nlm_list[nn,:], 'e')
    
        # Set eigen vectors
        e = np.zeros((3,3))
        e[0,:],e[1,:] = v1,np.array([-v1[2],0,v1[0]])
        e[2,:] = np.cross(e[0,:],e[1,:])
    
        # Calculate structure tensors
        a2, a4, a6, a8  = sf.ai(nlm_list[nn,:])
        a2t,a4t,a6t,a8t = sf.ai(nlm_list_trunc[nn,:])
    
        # Calculate enhancement factors
    
        tau0 = 1 # Stress function magnitude (does not matter, cancels in division)
    
        # Compressive enhancements
        for ii in np.arange(3):
            
            ei = e[ii,:]
            vw  = np.tensordot(ei,ei, axes=0)
            tau = tau0*(np.identity(3)/3 - vw) 
            
            Eii_true[nn,ii]  = sf.enhfac_vw(a2,a4,a6,a8,     vw,tau, Ecc,Eca,g,nprime)
            Eii_trunc[nn,ii] = sf.enhfac_vw(a2t,a4t,a6t,a8t, vw,tau, Ecc,Eca,g,nprime)
            
        # Shear enhancements
        for kk, (ii,jj) in enumerate([(0,1),(0,2),(1,2)]):
            
            ei = e[ii,:]
            ej = e[jj,:]
            vw  = np.tensordot(ei,ej, axes=0)
            tau = tau0*(vw+vw.T)
            
            Eij_true[nn,kk]  = sf.enhfac_vw(a2,a4,a6,a8,     vw,tau, Ecc,Eca,g,nprime)
            Eij_trunc[nn,kk] = sf.enhfac_vw(a2t,a4t,a6t,a8t, vw,tau, Ecc,Eca,g,nprime)
        
    ### Plot enhancement factors    
        
    pls   = ['-','--',':']
    c1,c2 = 'k','#b15928'
    
    for kk in np.arange(3):
        ii = kk
        ax_E.plot(np.divide(Eii_trunc[:,ii],Eii_true[:,ii]), zkm[1:], ls=pls[kk], color='k', lw=lw, label=r'$%i,%i$'%(ii+1,ii+1))
        
    for kk, (ii,jj) in enumerate([(0,1),(0,2),(1,2)]):
        ax_E.plot(np.divide(Eij_trunc[:,kk],Eij_true[:,kk]), zkm[1:], color='g', ls=pls[kk], lw=lw, label=r'$%i,%i$'%(ii+1,jj+1))
    
    hleg = ax_E.legend(loc=1,  bbox_to_anchor=(1.4,1), title='$p,q=$', **legkwargs)
    hleg.get_frame().set_linewidth(0.7)
    
    ticks = np.arange(0.5,2+1e-5, 0.25)
    ax_E.set_xticks(ticks[0::2])
    ax_E.set_xticks(ticks, minor=True)
    ax_E.set_xlim([0.6, 1.85])
    ax_E.set_xlabel(r'$E_{pq}(\psi^\dagger)/E_{pq}(\psi)$')
    
    writeSubplotLabel(ax_E, 2, r'{\bf f}', frameon=frameon, alpha=1.0, fontsize=FS)
    plt.setp(ax_E.get_yticklabels(), visible=False)
    setcb(ax_E, 0, phantom=True)
    
    #--------------------
    
    # Save plot
    fout = 'diagnostic_fabtype%i.png'%(FABRIC_TYPE)
    print('Saving %s'%(fout))
    plt.savefig(fout, dpi=300)