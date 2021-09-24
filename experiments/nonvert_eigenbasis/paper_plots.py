#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2021

import numpy as np
import copy, sys, os, code # code.interact(local=locals())
sys.path.insert(0, '../../lib')

from specfabpy import specfabpy as sf

from layer import *
from layerstack import *
from plottools import *

#---------------------
# Config
#---------------------

### Layer stack config

FABRIC_TYPE   = 1 # 1 = single max, 2 = girdle
DO_DIAGNOSTIC = 0 # Plot truncated profile results? ($\psi^\dagger$ profile)

DEBUG         = 1 # If 1, then plot in lower resolution

RESMUL = 1.5 if DEBUG else 6
N_layers = int(50*RESMUL) + 1 # Number of layers
N_frames = int(20*RESMUL) # Number of rotated frames of the radar system 

H = 2000 # Ice column height
z = np.linspace(0,-H, N_layers) # Interface positions

### Radar config

alpha  = np.deg2rad(0)  # Angle of incidence for reference experiment (should be normal incidence, i.e. 0)
alphaX = np.deg2rad(10) # Angle of incidence of obliquely sounding experiment

E0    = 1.0e3   # Transmitted wave magnitude from surface layer
freq  = 179.0e6 # Transmitted frequency from surface layer 

### Transfer matrix model config

#mt = 'FP' # Fujita--Paren
mt = 'GTM' # General Transfer Matrix model for glaciology (Rathmann et al., 2021)

#---------------------
# Setup
#---------------------

### Specfab

nlm_len = sf.init(4) # ODF expansion series MUST be truncated at at least L=4.
nlm_list = np.zeros((N_layers,nlm_len), dtype=np.complex128) # This is the list of $\psi_l^m$ coefficients (here "n" is used in place of "psi").
nlm_list[:,0] = 1/np.sqrt(4*np.pi) # Normalized distribution

### Construct idealized fabric profiles

# Single max. at bottom
if FABRIC_TYPE == 1: 
    n20 = 0.2 * 3.0 
    n21 = 0
    n22 = 0
    
# Girdle at bottom
if FABRIC_TYPE == 2: 
    n20 = 0.15 
    n21 = 0
    n22 = -3/2 * np.sqrt(2/3) * n20

# Strengthen fabric profiles with depth
linvec = np.linspace(0.5,1, N_layers-1)
I1 = 1 # First subsurface layer (should be 1, else for debugging)
nlm_list[I1:,3] =  n20*linvec[(I1-1):] # n20
nlm_list[I1:,4] = +n21*linvec[(I1-1):] # n21p
nlm_list[I1:,5] = +n22*linvec[(I1-1):] # n22p
nlm_list[:,2] = -np.conj(nlm_list[:,4]) # n21m
nlm_list[:,1] = +np.conj(nlm_list[:,5]) # n22m
    
if 0 and DEBUG:
    print('a^(2) in first and last layer are:')
    print(nlm_to_c2(nlm_list[0,:]))
    print(nlm_to_c2(nlm_list[1,:]))
    print(nlm_to_c2(nlm_list[-1,:]))
    
    print('nlm in first and last last layer are:')
    print(nlm_list[1,:])
    print(nlm_list[-1,:])

# Rotate ODF profile toward horizontal plane
angles = np.deg2rad(np.linspace(0,-90, N_layers-1))
for nn,b in enumerate(angles,1):
    c, s = np.cos(b), np.sin(b)
    Qy = np.array(((c, 0, s), (0,1,0), (-s,0,c))) # Rotation matrix for rotations around y-axis.
    c2 = nlm_to_c2(nlm_list[nn,:]) 
    c2_rot = np.matmul(np.matmul(Qy, c2), Qy.T)
    nlm_list[nn,:6], lm_list = c2_to_nlm(c2_rot) # Set rotated ODF spectral coefs

# Debug, rotate also fabric in the horizontal plane with depth
if 0: 
    angles = np.deg2rad(np.linspace(0,90, N_layers-1))
    nlm_ref = nlm_list[-1,:].copy()
    for nn,b in enumerate(angles,1):
        c, s = np.cos(b), np.sin(b)
        Qz = np.array(((c, -s, 0), (s,c,0), (0,0,1))) # Rotation matrix for rotations around z-axis.
        c2 = nlm_to_c2(nlm_list[nn,:]) 
        #c2 = nlm_to_c2(nlm_ref) 
        c2_rot = np.matmul(np.matmul(Qz, c2), Qz.T)
        nlm_list[nn,:6], lm_list = c2_to_nlm(c2_rot) # Set rotated ODF spectral coefs
    
    
#---------------------
# Run model
#---------------------

### Reference profile model
    
lstack = LayerStack(nlm_list, z, N_frames=N_frames, modeltype=mt) # init layer stack
returns = lstack.get_returns(E0, f=freq, alpha=alpha) # get returns for radar config
Pm_HH,Pm_HV, dP_HH,dP_HV, c_HHVV, E_HH,E_HV = returns # unpack
eigvals, e1,e2,e3, a2 = lstack.get_eigenbasis() 

lm   = lstack.lm   
zkm  = lstack.z*1e-3

if DO_DIAGNOSTIC:
    
    # Truncated profile of reference model
    nlm_list_trunc = nlm_list.copy()
    nlm_list_trunc[:,np.array([2,4])] = 0
    
    lstack_trunc = LayerStack(nlm_list_trunc, z, N_frames=N_frames, modeltype=mt)
    returns_trunc = lstack_trunc.get_returns(E0, f=freq, alpha=alpha)
    Pm_HH_trunc,Pm_HV_trunc, dP_HH_trunc,dP_HV_trunc, c_HHVV_trunc, E_HH_trunc,E_HV_trunc = returns_trunc
    _, e1_trunc,e2_trunc,e3_trunc, a2_trunc = lstack_trunc.get_eigenbasis()

### Oblique incidence

if DO_DIAGNOSTIC:
    
    lstack_alphaX = LayerStack(nlm_list, z, N_frames=N_frames, modeltype=mt)
    returns_alphaX = lstack_alphaX.get_returns(E0, f=freq, alpha=alphaX)
    _,_, dP_HH_alphaX,_, _, _,_ = returns_alphaX
    
    # Truncated profile of obliquely incident model
    lstack_trunc_alphaX = LayerStack(nlm_list_trunc, z, N_frames=N_frames, modeltype=mt)
    returns_trunc_alphaX = lstack_trunc_alphaX.get_returns(E0, f=freq, alpha=alphaX)
    _,_, dP_HH_trunc_alphaX,_, _, _,_ = returns_trunc_alphaX


#--------------------
# Plot reference experiment
#--------------------   
        
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams, rc
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cartopy.crs as ccrs

I0 = 1 # Layer index of first layer to plot (if I0=1 then skip surface reflection in results)

(_, ax_ODFs, _,_,_,_,_, IplotODF, prj,geo,rot0) = plot_returns(lstack.z, returns, a2, eigvals, I0=I0)

# Plot relevant principal axis on first ODF plot
def plot_ei(ax, vec, geo, mrk='o', lbl=False, phi_rel=0, theta_rel=-18):
    
    theta,phi = getPolarAngles(vec)
    if 0<=phi<=180 and (theta<=0): theta,phi = getPolarAngles(-vec)
    color = '#08519c'
    ax.plot([phi],[theta], mrk, ms=7.0, markerfacecolor="None", markeredgecolor=color, markeredgewidth=1.1, transform=geo)
    
    # Determine if this is the SINGLEMAX or GIRDLE experiment...
    if lbl:
        if np.abs(vec[2])> 0.01: ax.text(rot0-phi+phi_rel, theta+theta_rel, r'$\vb{e}_1$', horizontalalignment='left', color=color, transform=geo)
        else:                    ax.text(phi-22,theta+8, r'$\vb{e}_3$', horizontalalignment='left', color=color, transform=geo)

def getPolarAngles(vec):
    x,y,z = vec
    phi   = np.rad2deg(np.arctan2(y,x))
    theta = 90 - np.rad2deg(np.arccos(z))
    return (theta, phi)

ei_ref = e1 if (FABRIC_TYPE == 1) else e3
plot_ei(ax_ODFs[0], ei_ref[IplotODF[0],:], geo, lbl=True) 
plot_ei(ax_ODFs[1], ei_ref[IplotODF[1],:], geo) 
plot_ei(ax_ODFs[2], ei_ref[IplotODF[2],:], geo) 

# Save plot
fout = 'syntheticprofile_fabtype%i.png'%(FABRIC_TYPE)
print('Saving %s'%(fout))
plt.savefig(fout, dpi=300)

#--------------------
# Plot setup
#--------------------

if DO_DIAGNOSTIC:
    
    lw = 1.6
    legkwargs = {'frameon':True, 'fancybox':False, 'edgecolor':'k', 'framealpha':0.85, 'ncol':1, 'handlelength':1.34, 'labelspacing':0.3}
    frameon = 1 # Frame on subplot labels?
    
    IaxODF = [0,1,2]
    IplotODF = [1, int(len(z)/2), -1] # time steps to plot ODF for
    
    scale = 0.35
    fig = plt.figure(figsize=(18*scale,13*scale))
    
    r = 0.28
    gs_master = gridspec.GridSpec(1, 2, width_ratios=[r, 1-r])
    gs_master.update(top=0.97, bottom=0.11, left=-0.04, right=1-0.08, wspace=0.065)
    
    gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_master[0], hspace=0.70)
    ax_ODFs = [fig.add_subplot(gs[ii, 0], projection=prj) for ii in IaxODF]
       
    gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_master[1], wspace=0.17, hspace=0.95, width_ratios=[1,1,1])
    ax_dP_HH        = fig.add_subplot(gs[0, 0])
    ax_dP_HH_alphaX = fig.add_subplot(gs[0, 1], sharey=ax_dP_HH)
    ax_E            = fig.add_subplot(gs[0, 2], sharey=ax_dP_HH)
    
    #--------------------
    
    plot_ODFs(ax_ODFs, IplotODF, nlm_list_trunc[IplotODF,:], lm, zkm, geo, rot0, ODFsymb=r'\psi^\dagger')
    ei_ref = e1_trunc if (FABRIC_TYPE == 1) else e3_trunc
    plot_ei(ax_ODFs[0], ei_ref[IplotODF[0],:], geo, lbl=True) 
    plot_ei(ax_ODFs[1], ei_ref[IplotODF[1],:], geo) 
    plot_ei(ax_ODFs[2], ei_ref[IplotODF[2],:], geo) 

    #--------------------
    
    vmin = -0.5; vmax = -vmin
    ticks = np.arange(vmin, vmax+1e-5, vmax/2)
    
    Z_alpha0 = dP_HH - dP_HH_trunc
    h_HH = plotRxMaps(ax_dP_HH, Z_alpha0, zkm, I0, vmin=vmin,vmax=vmax)
    writeSubplotLabel(ax_dP_HH, 2 ,r'{\bf d}',frameon=frameon, alpha=1.0, fontsize=FS)
    writeSubplotLabel(ax_dP_HH, 1 ,r'$\alpha=\SI{%i}{\degree}$'%(np.rad2deg(alpha)),frameon=frameon, alpha=1.0, fontsize=FS)
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
    h_HH = plotRxMaps(ax_dP_HH_alphaX, Z_alphaX, zkm, I0, vmin=vmin,vmax=vmax)
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
    
    # Note: subscripts "i,j" are "p,q" in the article text
    
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
    
    
    #--------------------
    
    if DO_DIAGNOSTIC:
        
        delta_dP_HH        = np.abs(dP_HH - dP_HH_trunc)
        delta_dP_HH_alphaX = np.abs(dP_HH_alphaX - dP_HH_trunc_alphaX)
        
        for p in [50,75,90,95,99]:
            print('percentile(dP_HH - dP_HH_trunc, %i) = %.4f (%.4f for alpha=%i)'%(p, np.percentile(delta_dP_HH[:], p), \
                                                                                   np.percentile(delta_dP_HH_alphaX[:], p), np.rad2deg(alphaX)))
