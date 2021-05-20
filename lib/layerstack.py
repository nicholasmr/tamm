# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2021

"""
This class represents a vertical stack of horizontally (x,y) homogeneous layers of anisotropic polycrystalline ice
"""

import copy, sys, code # code.interact(local=locals())

import numpy as np
import numpy.linalg as lag

from scipy.optimize import minimize

from layer import *
from layer_GTM import *

class LayerStack:
    
    
    #def __init__(self, n2m, z, alpha, beta, f=179.0e6, epsa=3.19, epsc=3.16, epsair=1.000589, sigma=1e-5, mu=1): 
    def __init__(self, n2m, z, alpha, beta, f=179.0e6, epsa=3.17, epsc=3.17-0.034, epsair=1.000589, sigma=1e-5, mu=1): 

        self.VERBOSE = 1
        
        self.dz = np.abs(np.diff(z))
        self.z = z

        self.alpha = alpha # angle of incidence [rad]
        self.beta  = beta # horizontal rotations [rad]
      
        self.N_layers,_ = n2m.shape # number of layers 
        self.N_frames   = len(self.beta) # number of rotated frames
        self.layerlist  = np.arange(self.N_layers)
        self.framelist  = np.arange(self.N_frames)
        
        #-------------
        
        self.mu = mu        
        self.f  = f
        self.epsa   = epsa
        self.epsc   = epsc
        self.epsair = epsair
        self.sigma  = sigma
        #
        self.epsiso = (2*self.epsa+self.epsc)/3 
        self.zeta  = np.sqrt(self.epsiso)*np.sin(self.alpha) 
       
        if len(self.dz) != len(n2m): print('*** ERROR: len(z) != 1+len(n2m)')
 
        # For GTM code (no need to calculate thse mutiple times)
        self.Delta1234 = np.array([[1,0,0,0],
                                   [0,0,1,0],
                                   [0,1,0,0],
                                   [0,0,0,1]])
    
        self.Delta1234_inv = exact_inv(self.Delta1234)  
       
        # Set up layers with prescribed fabric profile
        self.init_with_n2m(n2m)
        

    def init_with_n2m(self, n2m):
        
        if self.VERBOSE: print('*** Initializing stack: frames x layers = %i x %i ...'%(self.N_frames,self.N_layers))
        
        self.n2m        = n2m
        self.frm_n2m    = [self.rotate_frame(n2m, b) for b in self.beta]         
        self.frm_layers = [self.get_stack(self.frm_n2m[ff], self.zeta) for ff in self.framelist]
        self.ref_layers = self.frm_layers[0] # assume the first frame is the "lab" reference frame. 
        self.URD        = self.get_URD() # requires self.frm_layers

    def update_single_layer(self, n2m, nn):
        
        self.n2m[nn,:]  = n2m
        self.frm_n2m    = [self.rotate_frame(self.n2m, b) for b in self.beta] # easiest just to recalc. the rotation of all layers.
        for ff in self.framelist: self.frm_layers[ff][nn] = Layer(self.frm_n2m[ff][nn], self.dz[nn], self.f, self.zeta, self.epsa, self.epsc, self.sigma, self.mu)
        self.ref_layers = self.frm_layers[0] 
        self.URD        = self.get_URD() # Lazy approach: just recalculate M for entire stack (not just the updated layer)
        
        
    def get_stack(self, n2m, zeta):
        
        lstack = [Layer(n2m[nn], self.dz[nn], self.f, zeta, self.epsa, self.epsc, self.sigma, self.mu) for nn in self.layerlist]
        
        return lstack
        
    
    def rotate_frame(self, n2m, beta):
        
        rotmat = np.diag([np.exp(+1j*m*beta) for m in np.arange(-2,2+1)])
        n2m_rot = np.einsum('ij,nj->ni',rotmat,n2m)
        return n2m_rot

    
    def get_eigenframe(self):
        
        # Get eigen values and vectors for the reference stack/frame (assume the first frame is the lab frame, but dosn't matter for the eigenvalues).
        
        shp = (self.N_layers,3)
        e1, e2, e3 = np.zeros(shp), np.zeros(shp), np.zeros(shp)
        eigvals = np.zeros(shp)
        
        for nn in self.layerlist:
            l = self.ref_layers[nn] 
            (eigvals[nn,:], e1[nn,:], e2[nn,:], e3[nn,:]) = l.get_eigframe()

        return eigvals, e1,e2,e3
    
    
    def get_URD(self):

        D = np.zeros((self.N_frames,self.N_layers, 2,2), dtype=np.complex) # Downard (fwd) propagation
        U = np.zeros((self.N_frames,self.N_layers, 2,2), dtype=np.complex) # Upward (rev) propagation
        D[:,0,:,:] = np.eye(2) # identity for surface layer (zero propagation path)
        U[:,0,:,:] = np.eye(2) # identity for surface layer (zero propagation path)
        
        laylist = self.layerlist[:-1] # Reduced layer list: can not calculate reflection from last layer since the interface matrix depends on the perm/fabric of the next (unspecified) layer

        Mfwd_next = np.array([ [self.frm_layers[ff][nn+1].Mfwd for nn in laylist] for ff in self.framelist], dtype=np.complex)        
        Mrev_next = np.array([ [self.frm_layers[ff][nn+1].Mrev for nn in laylist] for ff in self.framelist], dtype=np.complex)
        
        A         = np.array([ [self.frm_layers[ff][nn  ].Ai     for nn in laylist] for ff in self.framelist], dtype=np.complex)
        A_inv     = np.array([ [self.frm_layers[ff][nn  ].Ai_inv for nn in laylist] for ff in self.framelist], dtype=np.complex)
        Anext     = np.array([ [self.frm_layers[ff][nn+1].Ai     for nn in laylist] for ff in self.framelist], dtype=np.complex)
        Anext_inv = np.array([ [self.frm_layers[ff][nn+1].Ai_inv for nn in laylist] for ff in self.framelist], dtype=np.complex)
        
        R, T, Trev = self.get_T_R(A,A_inv, Anext,Anext_inv)
        
        D_mul = np.einsum('fnij,fnjk->fnik', Mfwd_next, T)
        U_mul = np.einsum('fnij,fnjk->fnik', Trev, Mrev_next)
        
        for nn in self.layerlist[:-1]:
            D[:,nn+1,:,:] = np.einsum('fij,fjk->fik', D_mul[:,nn,:,:],     D[:,nn,:,:]) # F_{n}*E_{0}^{-,Tx} = E_{n}^{-,downward}
            U[:,nn+1,:,:] = np.einsum('fij,fjk->fik',     U[:,nn,:,:], U_mul[:,nn,:,:]) # B_{n}*E_{n}^{-,upward} = E_{0,Rx}^{-}

        # Total downwards-reflected-upwards transformation (self.N_frames,self.N_layers-1, 2,2)
        URD = np.einsum('fnij,fnjk->fnik',U[:,:-1,:,:],np.einsum('fnij,fnjk->fnik',R,D[:,:-1,:,:])) 
        
        return URD
    
    
    def get_T_R(self, Aprev,Aprev_inv, Anext,Anext_inv):
        
        ### Downward propergating wave (GTM code)
        
        Gamma     = np.einsum('fnij,fnjk->fnik', Aprev_inv, Anext) 
        GammaStar = np.einsum('ij,fnjk,kl->fnil', self.Delta1234_inv, Gamma, self.Delta1234) 
        
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
        
        R = np.array([[rpp,rsp], [rps,rss]], dtype=np.complex) 

        # Transmission coefficients
        tpp = np.nan_to_num(+np.divide(GammaStar[:,:,2,2],Denom))
        tss = np.nan_to_num(+np.divide(GammaStar[:,:,0,0],Denom))
        tps = np.nan_to_num(-np.divide(GammaStar[:,:,2,0],Denom))
        tsp = np.nan_to_num(-np.divide(GammaStar[:,:,0,2],Denom))

        T = np.array([[tpp, tsp], [tps, tss]], dtype=np.complex)
        
        ### Upward propergating wave
        
        Gamma_rev = np.einsum('fnij,fnjk->fnik', Anext_inv, Aprev)

        # Common denominator for all coefficients
        Denom_rev = np.multiply(Gamma_rev[:,:,2,2],Gamma_rev[:,:,3,3])  - np.multiply(Gamma_rev[:,:,2,3],Gamma_rev[:,:,3,2]) 

        # Transmission coefficients
        tpp_rev = np.nan_to_num(+np.divide(Gamma_rev[:,:,3,3],Denom_rev))
        tss_rev = np.nan_to_num(+np.divide(Gamma_rev[:,:,2,2],Denom_rev))
        tps_rev = np.nan_to_num(-np.divide(Gamma_rev[:,:,3,2],Denom_rev))
        tsp_rev = np.nan_to_num(-np.divide(Gamma_rev[:,:,2,3],Denom_rev))

        Trev = np.array([[tpp_rev, tsp_rev], [tps_rev, tss_rev]], dtype=np.complex)
        
        # Return re-ordered tensors for convention used in code (frame,layer, ...)
        return np.einsum('ijfn->fnij',R), np.einsum('ijfn->fnij',T), np.einsum('ijfn->fnij',Trev)


    #####
    # RADAR RETURNS
    #####
    
    def dB(self, amp): return 20 * np.log10(amp) 
    
    def get_returns(self, E0, Tx_pol=([1,0],), nn=None):
    #def get_returns(self, E0, Tx_pol=([1,0],[0,1]), nn=None):
        
        # Tx_pol:  polarization of transmitted wave H (x dir), V (y dir)
        E_Tx = np.array([np.array([E0*p[0], E0*p[1]], dtype=np.complex) for p in Tx_pol]) # Init downard propergating wave in layer 0
        E_Rx = self.get_Rx_all(E_Tx, nn=nn) # (frame,layer, Tx/Rx pair)
        
        E_HH = E_Rx[:,:,0] # H trans, H recv
        E_HV = E_Rx[:,:,1] # H trans, V recv       
        #E_VH = E_Rx[:,:,2] # V trans, H recv
        #E_VV = E_Rx[:,:,3] # V trans, V recv         
        
        E_HH_abs = np.abs(E_HH)
        E_HV_abs = np.abs(E_HV)
        #E_VV_abs = np.abs(E_VV)
        
        P_HH  = self.dB(E_HH_abs)
        Pm_HH = self.dB(E_HH_abs.mean(axis=0, keepdims=True))
        dP_HH = P_HH - Pm_HH
        
        P_HV  = self.dB(E_HV_abs)
        Pm_HV = self.dB(E_HV_abs.mean(axis=0, keepdims=True))
        dP_HV = P_HV - Pm_HV

        h,v = 0,1 # p,s comp        
        numer = np.multiply(self.URD[:,:,h,h], np.conjugate(self.URD[:,:,v,v]))
        denom = np.multiply(np.abs(self.URD[:,:,h,h]), np.abs(self.URD[:,:,v,v]))
        c_HHVV = np.divide(numer, denom)
        c_HHVV *= np.exp(1j*np.pi) # phase offset: c_HHVV is defined only up to an arbitrary phase shift (Jordan et al., 2019)
        
        return (np.squeeze(Pm_HH),np.squeeze(Pm_HV), \
                np.squeeze(dP_HH.T),np.squeeze(dP_HV.T), \
                np.squeeze(c_HHVV.T), \
                np.squeeze(E_HH.T),np.squeeze(E_HV.T))


    def get_Rx_all(self, E_Tx_list, nn=None):

        Nl = self.N_layers-1 if nn is None else 1
        E_Rx = np.zeros((self.N_frames, Nl, 2*len(E_Tx_list)), np.complex) 

        for jj, E_Tx in enumerate(E_Tx_list): # Transmitted polarization
            I = [0+2*jj, 1+2*jj]        
            if nn is None: E_Rx[:,:,I] = np.einsum('fnij,j->fni', self.URD[:,:,:,:], E_Tx) 
            else:          E_Rx[:,0,I] = np.einsum('fij,j->fi',   self.URD[:,nn,:,:], E_Tx) 
        
        return E_Rx


    #####
    # INVERSION
    #####


    def infer_n2m(self, E0, observed, symtype=1, method='BFGS',tol=1e-4):

        if symtype == 0: # no symmetries
            guessvec_nn = np.zeros((5), dtype=np.float) 
        
        if symtype == 1: # diagonal <c^2> 
            guessvec_nn = np.zeros((2), dtype=np.float) 
        
        n2m_infr = np.zeros((self.N_layers, 5), dtype=np.complex) 
        Pm_HH,Pm_HV, dP_HH,dP_HV, c_HHVV, E_HH,E_HV = observed  
        
        for nn in self.layerlist[1:]: # Assume first layer is known = isotropic
            if self.VERBOSE: print('*** Inferring n2m of layer %i of %i using %s with tol=%.1e ...' % (nn, self.N_layers ,method, tol))
            mm = nn-1
            observed_mm = (Pm_HH[mm],Pm_HV[mm], dP_HH[mm,:],dP_HV[mm,:], c_HHVV[mm,:], E_HH[mm,:],E_HV[mm,:])
            result = minimize(self.J, guessvec_nn, (nn, n2m_infr, E0, observed_mm, symtype), method=method, tol=tol, options={'disp':0}) # BFGS
            n2m_infr_nn = self.guessvector_to_n2m(result.x, symtype=symtype)
            n2m_infr[nn,:] = n2m_infr_nn
            self.update_single_layer(n2m_infr_nn, nn)
        
        #self.init_with_n2m(n2m_infr)
        
        return n2m_infr
    
    
    def J(self, guessvec_nn, nn, n2m_infr, E0, observed_mm, symtype):

        mm = nn-1 # with fabric guess for layer "nn", we can estimate the return for layer "mm" (the layer above) since the interface matrix "L" can be calculated.
        Pm_HH,Pm_HV, dP_HH,dP_HV, c_HHVV, E_HH,E_HV = observed_mm # unpack
        
        n2m_infr_nn = self.guessvector_to_n2m(guessvec_nn, symtype=symtype)
        n2m_infr[nn,:] = n2m_infr_nn
        self.update_single_layer(n2m_infr_nn, nn)
        Pm_HH_guess,Pm_HV_guess, dP_HH_guess,dP_HV_guess, c_HHVV_guess, E_HH_guess,E_HV_guess = self.get_returns(E0, nn=mm)
       
        ####
        
        # Full electric field anomaly (for debugging)
        J_dE = 0
        if 0:
            dE_HH = E_HH - E_HH_guess
            dE_HV = E_HV - E_HV_guess
            J_dE += np.dot(dE_HH, np.conj(dE_HH))*self.dz[0] 
            J_dE += np.dot(dE_HV, np.conj(dE_HV))*self.dz[0] 

        # Power anomaly
        J_dP = 0
        if 1:
            dP_HH_err = np.exp(dP_HH) - np.exp(np.nan_to_num(dP_HH_guess))
            dP_HV_err = np.exp(dP_HV) - np.exp(np.nan_to_num(dP_HV_guess))
            J_dP += np.dot(dP_HH_err, dP_HH_err)
            J_dP += np.dot(dP_HV_err, dP_HV_err)
                    
        # Covariance phase
        J_phi = 0
        if 0:
            gam_phi = 1e+0
            #gam_phi = 1e-10
            dphi = np.angle(c_HHVV) - np.angle(np.nan_to_num(c_HHVV_guess))
            J_phi += gam_phi * np.dot(dphi, dphi)*self.dz[0] 
            
        # Minimal power
        J_Pm = 0
        if 1:
            gam_Pm = 1e8 #if nn < 3 else 1e5
            #J_Pm += gam_Pm * np.power( np.exp(np.nan_to_num(Pm_HH_guess)), 2)
            #J_Pm += gam_Pm * np.power( np.exp(np.nan_to_num(Pm_HV_guess)), 2)
            J_Pm += gam_Pm * np.exp(np.nan_to_num(Pm_HH_guess))
            J_Pm += gam_Pm * np.exp(np.nan_to_num(Pm_HV_guess))
            
            #gam_n20 = 1e10
            #print('----',np.abs(n2m_infr_nn[2]))
            #ll = gam_n20 * np.exp(1/(1e-1+np.abs(n2m_infr_nn[2]))) #1/(1e-3+np.abs(n2m_infr_nn[2]))
            #print(ll)
            #J_Pm +=  ll
            
            
        # Regularization (laplacian)
        J_reg_tikh = 0
        if 0:
            gam_tikh = 1e-5
            J_reg_tikh = gam_tikh * np.dot(n2m_infr_nn,np.conj(n2m_infr_nn)) 
       
         # Regularization (laplacian)
        J_reg_lapl = 0
        if 0:
            gam_lapl = 1e-4
            lapl_nn = 0 if nn < 2 else (n2m_infr[nn,:] - 2*n2m_infr[nn-1,:] + n2m_infr[nn-2,:])/self.dz[0]**2 # laplacian
            J_reg_lapl = gam_lapl * np.dot(lapl_nn, np.conj(lapl_nn))*self.dz[0]
        
        # Total
        J_reg = J_reg_tikh + J_reg_lapl
        J = J_dE + J_dP + J_phi + J_Pm + J_reg

#        print('(J, J_dE, J_dP, J_phi, J_reg) = (%.2e, %.2e, %.2e, %.2e, %.2e)'%(J, J_dE, J_dP, J_phi, J_reg))
       
        
        return J

    def guessvector_to_n2m(self, guessvec, symtype=0):
        
        if symtype == 0:
            n22m = guessvec[0] + 1j*guessvec[1]
            n21m = guessvec[2] + 1j*guessvec[3]
            n20  = guessvec[4]
        
        if symtype == 1:
            n22m = guessvec[0] + 0j
            n21m = 0 + 0j
            n20  = guessvec[1]
        
        n2m = np.array([n22m, n21m, n20, -np.conjugate(n21m), np.conjugate(n22m)], dtype=np.complex)

        return n2m 
