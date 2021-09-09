# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2021

"""
This class represents a vertical stack of horizontally (x,y) homogeneous layers of anisotropic polycrystalline ice
"""

import numpy as np

from layer import *
from layer_GTM import *

class LayerStack:
    
    def __init__(self, nlm, z, N_frames=100, epsa=[3.17], epsc=[3.17-0.034], sigma=[1e-5], mu=1, modeltype='GTM', VERBOSE=1): 

        self.modeltype = modeltype # GTM (General Transfer Matrix, aka. tamm) or FP (Fujita--Paren)
        if self.modeltype not in ['GTM','FP']: print('*** ERROR: allowed "modeltype" is "GTM" or "FP"')
        
        # Dimensions
        self.N_layers,_ = nlm.shape # number of layers 
        self.N_frames   = N_frames  # number of rotated frames
        self.layerlist  = np.arange(self.N_layers)
        self.framelist  = np.arange(self.N_frames)
        self.beta       = np.linspace(0, np.pi, self.N_frames) # horizontal rotations [rad]
        #
        self.z  = np.hstack(([np.abs(z[1]-z[0])],z)) # Add isotropic surface layer with thickness equal to first subsurface layer 
        self.dz = np.abs(np.diff(self.z))
     
        # Input parameters
        self.mu    = mu  
        self.epsa  = np.full((self.N_layers), epsa[0]) if len(epsa)==1 else epsa 
        self.epsc  = np.full((self.N_layers), epsc[0]) if len(epsc)==1 else epsc 
        self.sigma = np.full((self.N_layers), sigma[0]) if len(sigma)==1 else sigma 
        
        # Set up layers with prescribed fabric profile
        self.n2m = np.array([ nlm[nn, np.array([1,2,3,4,5])]/nlm[nn,0] for nn in np.arange(self.N_layers) ]) # normalized l=2 spectral coefs
        self.lm  = np.array([(2,-2),(2,-1),(2,0),(2,1),(2,2)]).T
        
        # Errors, warnings and verbosity
        
        if len(self.dz) != len(self.n2m): raise ValueError('len(z) != 1+len(n2m)')
        
        if np.any(nlm[0,1:]): raise ValueError('n_2^m of top layer must vanish (surface layer must be isotropic)')
        
        self.VERBOSE = VERBOSE
        if self.VERBOSE: print('Initialized "%s" stack: frames x layers = %i x %i'%(self.modeltype, self.N_frames,self.N_layers))

    
    def get_eigenframe(self):
        
        # Get eigen values and vectors for the reference stack/frame (assume the first frame is the lab frame, but of course does not matter for the eigenvalues).
        
        shp = (self.N_layers,3)
        e1, e2, e3 = np.zeros(shp), np.zeros(shp), np.zeros(shp)
        eigvals    = np.zeros(shp)
        a2         = np.zeros((self.N_layers,3,3))

        generic_layer = Layer([0,0,0,0,0], 0, 1, 1e3, 0, 1, 1, 0, 1) # pseudo layer to use the get_eigframe() method of the layer class
        
        for nn in self.layerlist:
            generic_layer.set_fabric(self.n2m[nn,:], 0)
            (eigvals[nn,:], e1[nn,:], e2[nn,:], e3[nn,:]) = generic_layer.get_eigframe()
            a2[nn,:,:] = generic_layer.ccavg # a^(2) = <c^2>

        return eigvals, e1,e2,e3, a2
    
    
    def get_stack(self, beta, f, zeta):

        lstack = [Layer(self.n2m[nn], beta, self.dz[nn], f, zeta, self.epsa[nn], self.epsc[nn], self.sigma[nn], self.mu, modeltype=self.modeltype) for nn in self.layerlist]
        return lstack


    def get_propconst(self):
    
        ks = np.array([self.frm_layers[0][nn].ks[:] for nn in self.layerlist[:]], dtype=np.complex128) # dimensionless propagation constants: k_s = omega/c q_s
        return ks
   
    
    #####
    # SYSTEM MATRICES
    #####
    

    def get_T_R(self, Aprev,Aprev_inv, Anext,Anext_inv):
        
        ### Downward propergating wave (original GTM code)
       
        Delta1234 = np.array([[1,0,0,0],
                              [0,0,1,0],
                              [0,1,0,0],
                              [0,0,0,1]])
    
        Delta1234_inv = exact_inv(Delta1234)  
        
        Gamma     = np.einsum('fnij,fnjk->fnik', Aprev_inv, Anext) 
        GammaStar = np.einsum('ij,fnjk,kl->fnil', Delta1234_inv, Gamma, Delta1234) 
        
        # Common denominator for all coefficients
        Denom = np.multiply(GammaStar[:,:,0,0],GammaStar[:,:,2,2]) - np.multiply(GammaStar[:,:,0,2],GammaStar[:,:,2,0])
        
        # Reflection coefficients
        rpp = np.multiply(GammaStar[:,:,1,0],GammaStar[:,:,2,2]) - np.multiply(GammaStar[:,:,1,2],GammaStar[:,:,2,0])
        rss = np.multiply(GammaStar[:,:,0,0],GammaStar[:,:,3,2]) - np.multiply(GammaStar[:,:,3,0],GammaStar[:,:,0,2])
        rps = np.multiply(GammaStar[:,:,3,0],GammaStar[:,:,2,2]) - np.multiply(GammaStar[:,:,3,2],GammaStar[:,:,2,0])
        rsp = np.multiply(GammaStar[:,:,0,0],GammaStar[:,:,1,2]) - np.multiply(GammaStar[:,:,1,0],GammaStar[:,:,0,2])

        rpp = np.nan_to_num(np.divide(rpp,Denom))
        rss = np.nan_to_num(np.divide(rss,Denom))
        rps = np.nan_to_num(np.divide(rps,Denom))        
        rsp = np.nan_to_num(np.divide(rsp,Denom))
        
        R = np.array([[rpp,rsp], [rps,rss]], dtype=np.complex128) 

        # Transmission coefficients
        tpp = np.nan_to_num(+np.divide(GammaStar[:,:,2,2],Denom))
        tss = np.nan_to_num(+np.divide(GammaStar[:,:,0,0],Denom))
        tps = np.nan_to_num(-np.divide(GammaStar[:,:,2,0],Denom))
        tsp = np.nan_to_num(-np.divide(GammaStar[:,:,0,2],Denom))

        T = np.array([[tpp, tsp], [tps, tss]], dtype=np.complex128)
        
        ### Upward propergating wave
        
        Gamma_rev = np.einsum('fnij,fnjk->fnik', Anext_inv, Aprev)

        # Common denominator for all coefficients
        Denom_rev = np.multiply(Gamma_rev[:,:,2,2],Gamma_rev[:,:,3,3])  - np.multiply(Gamma_rev[:,:,2,3],Gamma_rev[:,:,3,2]) 

        # Transmission coefficients
        tpp_rev = np.nan_to_num(+np.divide(Gamma_rev[:,:,3,3],Denom_rev))
        tss_rev = np.nan_to_num(+np.divide(Gamma_rev[:,:,2,2],Denom_rev))
        tps_rev = np.nan_to_num(-np.divide(Gamma_rev[:,:,3,2],Denom_rev))
        tsp_rev = np.nan_to_num(-np.divide(Gamma_rev[:,:,2,3],Denom_rev))

        Trev = np.array([[tpp_rev, tsp_rev], [tps_rev, tss_rev]], dtype=np.complex128)
        
        # Return re-ordered tensors for convention used in code (frame,layer, ...)
        return np.einsum('ijfn->fnij',R), np.einsum('ijfn->fnij',T), np.einsum('ijfn->fnij',Trev)

        
    def get_URD(self, frm_layers):

        D = np.zeros((self.N_frames,self.N_layers, 2,2), dtype=np.complex128) # Downward (fwd) propagation
        U = np.zeros((self.N_frames,self.N_layers, 2,2), dtype=np.complex128) # Upward (rev) propagation
        D[:,0,:,:] = np.eye(2) # identity for surface layer (zero propagation path)
        U[:,0,:,:] = np.eye(2) # identity for surface layer (zero propagation path)
        
        laylist = self.layerlist[:-1] # Reduced layer list: can not calculate reflection from last layer since the interface matrix depends on the perm/fabric of the next (unspecified) layer

        Mfwd_next = np.array([ [frm_layers[ff][nn+1].Mfwd for nn in laylist] for ff in self.framelist], dtype=np.complex128)        
        Mrev_next = np.array([ [frm_layers[ff][nn+1].Mrev for nn in laylist] for ff in self.framelist], dtype=np.complex128)

        if self.modeltype == 'GTM':
            
            A         = np.array([ [frm_layers[ff][nn  ].Ai     for nn in laylist] for ff in self.framelist], dtype=np.complex128)
            A_inv     = np.array([ [frm_layers[ff][nn  ].Ai_inv for nn in laylist] for ff in self.framelist], dtype=np.complex128)
            Anext     = np.array([ [frm_layers[ff][nn+1].Ai     for nn in laylist] for ff in self.framelist], dtype=np.complex128)
            Anext_inv = np.array([ [frm_layers[ff][nn+1].Ai_inv for nn in laylist] for ff in self.framelist], dtype=np.complex128)
            
            R, T, Trev = self.get_T_R(A,A_inv, Anext,Anext_inv)
            
        if self.modeltype == 'FP':
    
            R    = np.zeros((self.N_frames,len(laylist), 2,2), dtype=np.complex128)
            T    = np.zeros((self.N_frames,len(laylist), 2,2), dtype=np.complex128)
            Trev = np.zeros((self.N_frames,len(laylist), 2,2), dtype=np.complex128)
            
            # Transmission matrices = identity matrices
            T[:,:,0,0] = 1
            T[:,:,1,1] = 1
            Trev = T.copy()
            
            # Reflection matrices are based on Paren (1981) scattering
            ff = 0 # any beta frame will do (eigenvalues are not changed by rotation the frame of reference)
            R0flat = np.array([[ (frm_layers[ff][nn].ai_horiz[ii] - frm_layers[ff][nn+1].ai_horiz[ii])*(self.epsc[nn]-self.epsa[nn]) for ii in (0,1)] for nn in laylist], dtype=np.complex128)
            #R0flat *= 1/0.03404121647756628 # tamm cal.
            # ...reshape
            R0 = np.zeros((len(laylist), 2,2), dtype=np.complex128)
            R0[:,0,0] = +R0flat[:,0]
            R0[:,1,1] = +R0flat[:,1]
            # ...construct reflection matrices for each frame by a simple rotation around the z-axis or the reference frame
            for ff,b in enumerate(self.beta):
                for nn in self.layerlist[:-1]:
                    R[ff,nn,:,:] = frm_layers[ff][nn+1].eig_to_meas_frame(R0[nn,:,:])
        
        #-------------------
        
        D_mul = np.einsum('fnij,fnjk->fnik', Mfwd_next, T)
        U_mul = np.einsum('fnij,fnjk->fnik', Trev, Mrev_next)
        
        for nn in self.layerlist[:-1]:
            D[:,nn+1,:,:] = np.einsum('fij,fjk->fik', D_mul[:,nn,:,:],     D[:,nn,:,:]) # D_{n}*E_{0}^{-,Tx} = E_{n}^{-,downward}
            U[:,nn+1,:,:] = np.einsum('fij,fjk->fik',     U[:,nn,:,:], U_mul[:,nn,:,:]) # U_{n}*E_{n}^{-,upward} = E_{0,Rx}^{-}

        # Total downwards-reflected-upwards transformation (self.N_frames,self.N_layers-1, 2,2)
        URD = np.einsum('fnij,fnjk->fnik',U[:,:-1,:,:],np.einsum('fnij,fnjk->fnik',R,D[:,:-1,:,:])) 
        
        return URD
    

    #####
    # RADAR RETURNS
    #####
    
    
    def dB(self, amp): return 20 * np.log10(amp) 

   
    def get_returns(self, E0, f=179.0e6, alpha=0, nn=None, plot=False):
    
        # Construct URD matrix for total two-way propagation
        self.epsiso = (2*self.epsa[0]+self.epsc[0])/3 # isotropic permitivity of surface (top) layer
        zeta  = np.sqrt(self.epsiso)*np.sin(alpha) # Dimensionless x-component of incident wave vector
        frm_layers = [self.get_stack(b, f, zeta) for b in self.beta]
        
        URD = self.get_URD(frm_layers) 
    
        # Tx_pol:  polarization of transmitted wave H (x dir), V (y dir)
        Tx_pol = ([1,0],)
        E_Tx = np.array([np.array([E0*p[0], E0*p[1]], dtype=np.complex128) for p in Tx_pol]) # Init downard propergating wave in layer 0
        E_Rx = self.get_Rx_all(E_Tx, URD, nn=nn) # (frame, layer, Tx/Rx pair)
        
        E_HH = E_Rx[:,:,0] # Tx,Rx = H,H
        E_HV = E_Rx[:,:,1] # Tx,Rx = H,V
        #E_VH = E_Rx[:,:,2] # Tx,Rx = V,H
        #E_VV = E_Rx[:,:,3] # Tx,Rx = V,V         
        
        E_HH_abs = np.abs(E_HH)
        E_HV_abs = np.abs(E_HV)
        #E_VV_abs = np.abs(E_VV)
        
        P_HH  = self.dB(E_HH_abs)
        Pm_HH = self.dB(E_HH_abs.mean(axis=0, keepdims=True))
        dP_HH = P_HH - Pm_HH
        
        P_HV  = self.dB(E_HV_abs)
        Pm_HV = self.dB(E_HV_abs.mean(axis=0, keepdims=True))
        dP_HV = P_HV - Pm_HV

        h,v = 0,1 # p (H), s (V) components
        I = self.layerlist[:-1] if nn==None else np.array([nn]) # Consider only the requested layer (nn) ?
        numer = np.multiply(       URD[:,I,h,h],  np.conjugate(URD[:,I,v,v]))
        denom = np.multiply(np.abs(URD[:,I,h,h]),       np.abs(URD[:,I,v,v]))
        c_HHVV = np.divide(numer, denom)
        
        # phase offset: c_HHVV is defined only up to an arbitrary phase shift <=> exp[i*(phi1+const.)]*exp[-i*(phi2+const.)] = exp[i*(phi1-phi2)]
        if self.modeltype=='GTM':
            c_HHVV *= np.exp(1j*np.pi) 
        
        returns = (np.squeeze(Pm_HH),np.squeeze(Pm_HV), 
                np.squeeze(dP_HH.T),np.squeeze(dP_HV.T), \
                np.squeeze(c_HHVV.T), \
                np.squeeze(E_HH.T),np.squeeze(E_HV.T))

        if plot: self.plot(returns)
        
        # Save for later use (e.g. diagnostics)
        self.frm_layers = frm_layers 
        
        return returns


    def get_Rx_all(self, E_Tx_list, URD, nn=None):

        Nl = self.N_layers-1 if nn is None else 1
        E_Rx = np.zeros((self.N_frames, Nl, 2*len(E_Tx_list)), np.complex128) 

        for jj, E_Tx in enumerate(E_Tx_list): # Transmitted polarization
            I = [0+2*jj, 1+2*jj]        
            if nn is None: E_Rx[:,:,I] = np.einsum('fnij,j->fni', URD[:,:,:,:],  E_Tx) 
            else:          E_Rx[:,0,I] = np.einsum('fij,j->fi',   URD[:,nn,:,:], E_Tx) 
        
        return E_Rx
    

    #####
    # PLOT
    #####    
    
    def plot(self, returns):
        
        import matplotlib.pyplot as plt
        from matplotlib import rcParams, rc
        from matplotlib.offsetbox import AnchoredText
        import matplotlib.gridspec as gridspec
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        FS = 12
        rc('font',**{'family':'serif','sans-serif':['Times'],'size':FS})
        rc('text', usetex=True)
        rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{physics} \usepackage{txfonts} \usepackage{siunitx}'

        Pm_HH,Pm_HV, dP_HH,dP_HV, c_HHVV, E_HH,E_HV = returns
        zkm = 1e-3 * self.z
        
        #--------------------
        # Plot
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
