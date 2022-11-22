#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2021-2022

import numpy as np
import copy, sys, os, code # code.interact(local=locals())

sys.path.insert(0, '../../lib')
from layer import *
from layerstack import *
from plottools import *

sys.path.insert(0, '../demo')
from specfabpy import specfabpy as sf
from synthfabric import * 

#---------------------
# Config
#---------------------

### Layer stack config

"""
    FABRIC_TYPE:
    ------------
    0 = Single max., vertical eigen frame (for debugging/comparing GTM and FP implementations)
    1 = Single max., vertical rotation 
    2 = Single max., vertical+horizontal rotation
    3 = Girdle, vertical rotation
    4 = Girdle, vertical+horizontal rotation
    5 = Shear margin (horizontal shear in top part, vertical shear dominated by DDRX near bottom)
    6 = Bed bump (vertical shear in top part, horizontal compression near bottom)
"""

FABRIC_TYPES = [1,2,3,4,5,6]
#FABRIC_TYPES = [0] # debug

DO_OBLIQUE   = 1 # Plot truncated profile results? ($\psi^\dagger$ profile)
DEBUG        = 0 # If 1, then plot in lower resolution


RESMUL = 1.5 if DEBUG else 5
N_layers = int(50*RESMUL) + 1 # Number of layers
N_frames = int(20*RESMUL) # Number of rotated frames of the radar system 

H = 2000 # Ice column height
z = np.linspace(0,-H, N_layers) # Interface positions

### Radar config

alpha  = np.deg2rad(0)  # Angle of incidence for reference experiment (should be normal incidence, i.e. 0)
alphaX = np.deg2rad(10) # Angle of incidence of obliquely sounding experiment

E0    = 1.0e3   # Transmitted wave magnitude from surface layer
freq  = 179.0e6 # Transmitted frequency from surface layer 

### Transfer matrix model type

#modeltype = 'FP'  # Fujita--Paren model
modeltype = 'GTM' # General Transfer Matrix model (Rathmann et al., 2021)

### Fabric model 

L = 12 # Trancation of ODF expansion series
        
############################################
        
for FABRIC_TYPE in FABRIC_TYPES:

    FABRIC_SINGLEMAX   = (FABRIC_TYPE==0 or FABRIC_TYPE==1 or FABRIC_TYPE==2)
    FABRIC_GIRDLE      = (FABRIC_TYPE==3 or FABRIC_TYPE==4)
    FABRIC_FROMSPECFAB = (FABRIC_TYPE==5 or FABRIC_TYPE==6)

    #---------------------
    # Fabric profile
    #---------------------

    ### Synthetic profiles from idealized <c^2> profiles.
    if FABRIC_SINGLEMAX or FABRIC_GIRDLE:

        ### Specfab

        lm, nlm_len = sf.init(L) 
        nlm_list = np.zeros((N_layers,nlm_len), dtype=np.complex128) # This is the list of $\psi_l^m$ coefficients (here "n" is used in place of "psi").
        nlm_list[:,0] = 1/np.sqrt(4*np.pi) # Normalized distribution

        ### Construct idealized fabric profiles

        n20, n21, n22 = 0, 0, 0 # init

        if FABRIC_TYPE == 0:
            # Debug case: simple fabric profiles that allow verifying polarity of phase coherence (c_HHVV) against observations.
            if 0:
                # Young et al. (2021): synthetic fabric that approximates Fig. 6 in so that 
                #   this model approx. reproduces their FP model response (Fig. 5).
                n20 = 0.2 * 2
                n22 = 0.10 * np.exp(1j*np.pi)
            else:
                # Ershadi et al. (2021): synthetic fabric that approximates Fig. 5 in so that 
                #   this model approx. reproduces their FP model response (Fig. 5).
                n20 = 0.2 * 2.2
                n22 = 0.05 * np.exp(1j*np.pi/2)
            linvec = np.linspace(0.3,1, N_layers-1)
        
        if FABRIC_TYPE == 1 or FABRIC_TYPE == 2: 
            n20 = 0.2 * 3
            n22 = 1e-10 # Set small non-zero value that removes degeneracy (no effect on results) and allows enhancement-factor plots to be well-behaved.
            linvec = np.linspace(0.5,1, N_layers-1)
                    
        if FABRIC_TYPE == 3 or FABRIC_TYPE == 4: 
            n20 = 0.15 
            n22 = -3/2 * np.sqrt(2/3) * n20
            n22 += 1e-10 # Set small non-zero value that removes degeneracy (no effect on results) and allows enhancement-factor plots to be well-behaved.
            linvec = np.linspace(0.5,1, N_layers-1)

        # Strengthen fabric profiles with depth
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

        if FABRIC_TYPE > 0: 
            
            # Rotate ODF profile toward horizontal plane
            angles = np.deg2rad(np.linspace(0,-90, N_layers-1))
            for nn,b in enumerate(angles,1):
                c, s = np.cos(b), np.sin(b)
                Qy = np.array(((c, 0, s), (0,1,0), (-s,0,c))) # Rotation matrix for rotations around y-axis.
                c2 = nlm_to_c2(nlm_list[nn,:]) 
                c2_rot = np.matmul(np.matmul(Qy, c2), Qy.T)
                nlm_list[nn,:6], lm_list = c2_to_nlm(c2_rot) # Set rotated ODF spectral coefs

            # Rotate additionally the ODF profile in the horizontal plane with depth
            if FABRIC_TYPE == 2 or FABRIC_TYPE == 4: 
                angles = np.deg2rad(np.linspace(0,90, N_layers-1))
                nlm_ref = nlm_list[-1,:].copy()
                for nn,b in enumerate(angles,1):
                    c, s = np.cos(b), np.sin(b)
                    Qz = np.array(((c, -s, 0), (s,c,0), (0,0,1))) # Rotation matrix for rotations around z-axis.
                    c2 = nlm_to_c2(nlm_list[nn,:]) 
                    c2_rot = np.matmul(np.matmul(Qz, c2), Qz.T)
                    nlm_list[nn,:6], lm_list = c2_to_nlm(c2_rot) # Set rotated ODF spectral coefs

    ### Load fabric profiles generated externally using specfab       
    else:
    
        synfab = SyntheticFabric(H=H, N=N_layers, L=L) 

        if FABRIC_TYPE==5:
            # Shear-margin-like
            mode1 = {'plane':'yx', 't_c':-150, 'Gamma0':0}
            mode2 = {'plane':'zx', 't_c':500, 'Gamma0':1.5e-9}
            nlm_list, lm_true, z = synfab.make_profile(mode1, mode2, dt=5, t_end=1000, crossover=[0.5,0.8], plot=False)
        
        if FABRIC_TYPE==6: 
            # Ice-stream-like
            mode1 = {'plane':'zx', 't_c':100,         'Gamma0':0}
            mode2 = {   'ax':'x',  't_c':+200, 'r':0, 'Gamma0':0}
            nlm_list, lm_true, z = synfab.make_profile(mode1, mode2, dt=5, t_end=1000, crossover=[0.6,0.8], plot=False) 

    #---------------------
    # Run model
    #---------------------

    ### Reference profile model
        
    lstack = LayerStack(nlm_list, z, N_frames=N_frames, modeltype=modeltype) # init layer stack
    returns = lstack.get_returns(E0, f=freq, alpha=alpha) # get returns for radar config
    Pm_HH,Pm_HV, dP_HH,dP_HV, c_HHVV, E_HH,E_HV = returns # unpack
    eigvals, e1,e2,e3, a2 = lstack.get_eigenbasis() 

    lm   = lstack.lm   
    zkm  = lstack.z*1e-3

    if DO_OBLIQUE:
        
        # Truncated profile of reference model
        nlm_list_trunc = nlm_list.copy()
        nlm_list_trunc[:,np.array([2,4])] = 0
        
        lstack_trunc = LayerStack(nlm_list_trunc, z, N_frames=N_frames, modeltype=modeltype)
        returns_trunc = lstack_trunc.get_returns(E0, f=freq, alpha=alpha)
        Pm_HH_trunc,Pm_HV_trunc, dP_HH_trunc,dP_HV_trunc, c_HHVV_trunc, E_HH_trunc,E_HV_trunc = returns_trunc
        _, e1_trunc,e2_trunc,e3_trunc, a2_trunc = lstack_trunc.get_eigenbasis()

    ### Oblique incidence

    if DO_OBLIQUE:
        
        lstack_alphaX = LayerStack(nlm_list, z, N_frames=N_frames, modeltype=modeltype)
        returns_alphaX = lstack_alphaX.get_returns(E0, f=freq, alpha=alphaX)
        _,_, dP_HH_alphaX,dP_HV_alphaX, c_HHVV_alphaX, _,_ = returns_alphaX
        
        # Truncated profile of obliquely incident model
        lstack_trunc_alphaX = LayerStack(nlm_list_trunc, z, N_frames=N_frames, modeltype=modeltype)
        returns_trunc_alphaX = lstack_trunc_alphaX.get_returns(E0, f=freq, alpha=alphaX)
        _,_, dP_HH_trunc_alphaX,dP_HV_trunc_alphaX, c_HHVV_trunc_alphaX, _,_ = returns_trunc_alphaX


    #--------------------
    # Plot 1 of 2 (synthetic*)
    #--------------------   
            
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import rcParams, rc
    from matplotlib.offsetbox import AnchoredText
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import cartopy.crs as ccrs

    I0 = 1 # Layer index of first layer to plot (if I0=1 then skip surface reflection in results)
    kwargs = {'nlm_true':nlm_list, 'lm_true':lm_true} if FABRIC_FROMSPECFAB else {}
    (_, ax_ODFs, _,_,_,_,_, IplotODF, prj,geo,rot0) = plot_returns(lstack.z, returns, a2, eigvals, I0=I0, tickintvl=8, **kwargs)

    # Plot principal ODF axis
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

    if FABRIC_TYPE==1: 
        ei_ref = e1 if FABRIC_SINGLEMAX else e3
        plot_ei(ax_ODFs[0], ei_ref[IplotODF[0],:], geo, lbl=True) 
        plot_ei(ax_ODFs[1], ei_ref[IplotODF[1],:], geo) 
        plot_ei(ax_ODFs[2], ei_ref[IplotODF[2],:], geo) 

    # Save plot
    fout = 'synthetic%i-%s.png'%(FABRIC_TYPE, modeltype)
    print('Saving %s'%(fout))
    plt.savefig(fout, dpi=300)

    #--------------------
    # Plot 2 of 2 (diagnostic*)
    #--------------------

    PLOT_REDUCED = (FABRIC_TYPE==1) # Plot HV anomalies, too, unless the experiment is experiment 1 shown in the main text.

    if DO_OBLIQUE:
        
        lw = 1.6
        legkwargs = {'frameon':True, 'fancybox':False, 'edgecolor':'k', 'framealpha':0.85, 'ncol':1, 'handlelength':1.34, 'labelspacing':0.3}
        frameon = 1 # Frame on subplot labels?
        
        IaxODF = [0,1,2]
        IplotODF = [1, int(len(z)/2), -1] # time steps to plot ODF for
        
        scale = 0.35
        
        if PLOT_REDUCED:
            
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
            
        else:
                
            fig = plt.figure(figsize=(26*scale,13*scale))
            
            gs_master = gridspec.GridSpec(1, 2, width_ratios=[1/6, 5/6])
            gs_master.update(top=0.965, bottom=0.11, left=-0.0175, right=1-0.052, wspace=0.065)
            
            gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_master[0], hspace=0.70)
            ax_ODFs = [fig.add_subplot(gs[ii, 0], projection=prj) for ii in np.arange(3)]
               
            k = 0.9
            gs = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs_master[1], wspace=0.17, hspace=0.95, width_ratios=[k,k,1,1,1])
            ax_dP_HH        = fig.add_subplot(gs[0, 0])
            ax_dP_HH_alphaX = fig.add_subplot(gs[0, 1], sharey=ax_dP_HH)
            ax_dP_HV        = fig.add_subplot(gs[0, 2], sharey=ax_dP_HH)
            ax_dP_HV_alphaX = fig.add_subplot(gs[0, 3], sharey=ax_dP_HH)
            ax_E            = fig.add_subplot(gs[0, 4], sharey=ax_dP_HH)
        
        #--------------------
        
        plot_ODFs(ax_ODFs, nlm_list_trunc[IplotODF,:], lm_true if FABRIC_FROMSPECFAB else lm, zkm[IplotODF], geo, rot0, ODFsymb=r'\psi^\dagger', tickintvl=8)
        
        if FABRIC_TYPE==1:  
            ei_ref = e1_trunc if FABRIC_SINGLEMAX else e3_trunc
            plot_ei(ax_ODFs[0], ei_ref[IplotODF[0],:], geo, lbl=True) 
            plot_ei(ax_ODFs[1], ei_ref[IplotODF[1],:], geo) 
            plot_ei(ax_ODFs[2], ei_ref[IplotODF[2],:], geo) 

        #--------------------
        
        PLOT_COVVAR = 0 # DEBUG: test c_HV anomaly
        
        vmin = -0.5; vmax = -vmin; ticks = np.arange(vmin, vmax+1e-5, vmax/2)
        Z_alpha0 = dP_HH - dP_HH_trunc # test HH
        if PLOT_COVVAR:
            vmin = -90; vmax = -vmin; ticks = np.arange(vmin, vmax+1e-5, vmax/2)
            Z_alpha0 = np.angle(c_HHVV, deg=True) - np.angle(c_HHVV_trunc, deg=True) # test c_HV        
        ax = ax_dP_HH
        h_HH = plotRxMaps(ax, Z_alpha0, zkm, I0, vmin=vmin,vmax=vmax)
        writeSubplotLabel(ax, 2 ,r'{\bf d}',frameon=frameon, alpha=1.0, fontsize=FS)
        writeSubplotLabel(ax, 1 ,r'$\alpha=\SI{%i}{\degree}$'%(np.rad2deg(alpha)),frameon=frameon, alpha=1.0, fontsize=FS)
        setcb(ax, h_HH, ticks=ticks, xlbl=r'$\delta P_{\mathrm{HH}} - \delta P_{\mathrm{HH}}^\dagger$ (dB)')
        
        # Setup shared y-axis 
        da, daminor = 1, 0.1
        ax.set_yticks([0,-1,-2])
        ax.set_yticks(np.flipud(-np.arange(zkm[I0],5,daminor)), minor=True)
        ax.set_ylabel(r'z ($\SI{}{\kilo\metre}$)', fontsize=FS)
        ax.set_ylim(zkm[[-1,I0]])
        
        #--------------------

        vmin = -15; vmax = -vmin; ticks = np.arange(vmin, vmax+1e-5, vmax/2)
        Z_alphaX = dP_HH_alphaX - dP_HH_trunc_alphaX # test HH
        if PLOT_COVVAR:
            vmin = -90; vmax = -vmin; ticks = np.arange(vmin, vmax+1e-5, vmax/2)
            Z_alphaX = np.angle(c_HHVV_alphaX, deg=True) - np.angle(c_HHVV_trunc_alphaX, deg=True) # test c_HV
        ax = ax_dP_HH_alphaX
        h_HH = plotRxMaps(ax, Z_alphaX, zkm, I0, vmin=vmin,vmax=vmax)
        writeSubplotLabel(ax,2,r'{\bf e}',frameon=frameon, alpha=1.0, fontsize=FS)
        writeSubplotLabel(ax,1,r'$\alpha=\SI{%i}{\degree}$'%(np.rad2deg(alphaX)),frameon=frameon, alpha=1.0, fontsize=FS)
        setcb(ax, h_HH, ticks=ticks, xlbl=r'$\delta P_{\mathrm{HH}} - \delta P_{\mathrm{HH}}^\dagger$ (dB)')
        plt.setp(ax.get_yticklabels(), visible=False)
        
        #--------------------
        
        if not PLOT_REDUCED:

            vmin = -0.5; vmax = -vmin; ticks = np.arange(vmin, vmax+1e-5, vmax/2)
            Z_alpha0 = dP_HV - dP_HV_trunc # test HV
            ax = ax_dP_HV
            h_HH = plotRxMaps(ax, Z_alpha0, zkm, I0, vmin=vmin,vmax=vmax)
            writeSubplotLabel(ax, 2 ,r'{\bf f}',frameon=frameon, alpha=1.0, fontsize=FS)
            writeSubplotLabel(ax, 1 ,r'$\alpha=\SI{%i}{\degree}$'%(np.rad2deg(alpha)),frameon=frameon, alpha=1.0, fontsize=FS)
            setcb(ax, h_HH, ticks=ticks, xlbl=r'$\delta P_{\mathrm{HV}} - \delta P_{\mathrm{HV}}^\dagger$ (dB)')
            plt.setp(ax.get_yticklabels(), visible=False)     
                        
            #--------------------

            vmin = -15; vmax = -vmin; ticks = np.arange(vmin, vmax+1e-5, vmax/2)
            Z_alphaX = dP_HV_alphaX - dP_HV_trunc_alphaX # test HV
            ax = ax_dP_HV_alphaX
            h_HH = plotRxMaps(ax, Z_alphaX, zkm, I0, vmin=vmin,vmax=vmax)
            writeSubplotLabel(ax,2,r'{\bf g}',frameon=frameon, alpha=1.0, fontsize=FS)
            writeSubplotLabel(ax,1,r'$\alpha=\SI{%i}{\degree}$'%(np.rad2deg(alphaX)),frameon=frameon, alpha=1.0, fontsize=FS)
            setcb(ax, h_HH, ticks=ticks, xlbl=r'$\delta P_{\mathrm{HV}} - \delta P_{\mathrm{HV}}^\dagger$ (dB)')
            plt.setp(ax.get_yticklabels(), visible=False)     

        #--------------------

        ### Monocrystal rheological params
        
        # Use linear grain rheology from Rathmann and Lilien (2021)
        nprime = 1 
        Ecc    = sf.Ecc_opt_lin 
        Eca    = sf.Eca_opt_lin
        g      = sf.alpha_opt_lin
        
        ### Calculate enhancement factors 
        
        # Note: subscripts "i,j" are "p,q" in the article text

        Eij_true  = np.zeros((N_layers,3,3))
        Eij_trunc = np.zeros((N_layers,3,3))
        
        for nn in np.arange(N_layers)[:]:
            
            # Set eigen vectors
            e = np.zeros((3,3))
            e[0,:],e[1,:],e[2,:] = e1[nn,:],e2[nn,:],e3[nn,:]
       
            # Calculate enhancement factors
            for kk, (ii,jj) in enumerate([(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]):
                
                ei,ej = e[ii,:],e[jj,:]
                vw  = np.tensordot(ei,ej, axes=0)
                tau0 = 1 # Stress function magnitude (does not matter, cancels in division)
                tau = tau0*(np.identity(3)/3 - vw) if ii==jj else tau0*(vw+vw.T)
                
                Eij_true[nn,ii,jj]  = sf.Evw(nlm_list[nn,:],       vw,tau, Ecc,Eca,g,nprime)
                Eij_trunc[nn,ii,jj] = sf.Evw(nlm_list_trunc[nn,:], vw,tau, Ecc,Eca,g,nprime)
            
        ### Plot enhancement factors    
            
        pls   = ['-','--',':', '-','--',':']
        c1,c2 = 'k','#b15928'
        
        for kk, (ii,jj) in enumerate([(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]):
            ax_E.plot(np.divide(Eij_trunc[:,ii,jj],Eij_true[:,ii,jj]), zkm[1:], color='g' if kk>=3 else 'k', ls=pls[kk], lw=lw, label=r'$%i,%i$'%(ii+1,jj+1))
        
        hleg = ax_E.legend(loc=1,  bbox_to_anchor=(1.4,1), title='$p,q=$', **legkwargs)
        hleg.get_frame().set_linewidth(0.7)
        
        ticks = np.arange(0.5,2+1e-5, 0.25)
        ax_E.set_xticks(ticks[0::2])
        ax_E.set_xticks(ticks, minor=True)
        ax_E.set_xlim([0.6, 1.85])
        ax_E.set_xlabel(r'$E_{pq}(\psi^\dagger)/E_{pq}(\psi)$')
        
        writeSubplotLabel(ax_E, 2, r'{\bf %s}'%('f' if PLOT_REDUCED else 'h'), frameon=frameon, alpha=1.0, fontsize=FS)
        plt.setp(ax_E.get_yticklabels(), visible=False)
        setcb(ax_E, 0, phantom=True)
        
        #--------------------
        
        # Save plot
        fout = 'diagnostic%i-%s.png'%(FABRIC_TYPE, modeltype)
        print('Saving %s'%(fout))
        plt.savefig(fout, dpi=300)
        
        #--------------------
        
        if DO_OBLIQUE:
            
            delta_dP_HH        = np.abs(dP_HH - dP_HH_trunc)
            delta_dP_HH_alphaX = np.abs(dP_HH_alphaX - dP_HH_trunc_alphaX)
            
            for p in [50,75,90,95,99]:
                print('percentile(dP_HH - dP_HH_trunc, %i) = %.4f (%.4f for alpha=%i)'%(p, np.percentile(delta_dP_HH[:], p), \
                                                                                       np.percentile(delta_dP_HH_alphaX[:], p), np.rad2deg(alphaX)))
