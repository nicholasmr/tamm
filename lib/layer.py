# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2021

"""
This subclass wraps a spectral fabric formulation around the Passler and Paarmann (2017,2019) layer class implemented by M. Jeannin (2019)
"""

import numpy as np
import numpy.linalg as lag

from layer_PPJ import *

class Layer(Layer_PPJ):
    
    def __init__(self, nlm, beta, thickness, f, zeta, epsa, epsc, sigma, mu, modeltype='GTM'): 
        
        super().__init__()
        
        self.modeltype = modeltype # GTM (General Transfer Matrix) or FP (Fujita--Paren)
        if self.modeltype not in ['GTM','FP']: raise ValueError('Argument "modeltype" must be either "GTM" or "FP"')
        
        self.thick  = thickness # Layer thickness (d)
        self.f      = f         # Plane wave frequency 
        self.zeta   = zeta      # x-component of wave vector (dimensionless)
        
        self.epsa  = epsa  # Complex permitivity perdendicular to c-axis
        self.epsc  = epsc  # Complex permitivity parallel to c-axis
        self.sigma = sigma # Single crystal isotropic conductivity
        self.mu    = mu    # Relative permeability
        
        self.setup(nlm, beta) 

        
    def setup(self, nlm, beta):
        
        # Given the orientation fabric (nlm), all layer matrices are calculated by this method.
        # Call this method if you wish to update the layer with a new orientation fabric and re-calculate all layer properties (matrices etc.).
        
        # Fabric frame 
        self.nlm_0   = nlm # Save copy
        self.ccavg_0 = nlm_to_c2(self.nlm_0) # <c^2> in fabric frame
        
        # Measurement frame
        self.beta = beta
        Qz = np.diag([np.exp(+1j*m*self.beta) for m in [0, -2,-1,0,+1,+2]]) # Rotation matrix for rotations about the vertical z-axis.
        self.nlm = np.matmul(Qz, self.nlm_0) 
        self.ccavg = nlm_to_c2(self.nlm) # <c^2> in measurement frame
        
        # Calculate permitivity tensor and (transfer) matrices
        self.epsilon = self.get_epsavg(self.ccavg)  
        self.set_matrices() # Requires self.epsilon be set
        

    def get_epsavg(self, ccavg):
        
        monopole   = (2*self.epsa+self.epsc)/3 * np.eye(3)
        quadrupole = (self.epsc-self.epsa) * (ccavg - np.eye(3)/3)
        epsavg     = np.real(monopole+quadrupole) # Should be real; numerical errors might cause small non-zero imaginary parts.
        
        epsdprime = self.sigma/(2*np.pi*self.f*eps0) 
        epsavg = epsavg - 1j*epsdprime * np.eye(3) # Add isotropic imaginary part (due to acidity)
        
        return epsavg
    
    
    def set_matrices(self):
        
        ### Horizontal eigenvalues 
        
        self.ai_horiz, self.Qeig = lag.eig(self.ccavg_0[:2,:2])
        self.Qeiginv = self.Qeig.T # or lag.inv(self.Qeig)
        self.eigang = np.arctan2(self.Qeig[1,0],self.Qeig[0,0])
        
        ### Setup layer matrices
        
        # Short hands
        omega = 2*np.pi*self.f
        q2k = omega/c_const       
        
        # General Transfer Matrix
        if self.modeltype == 'GTM': 
        
            # Call parent GTM routines
            self.calculate_matrices(self.zeta)
            self.calculate_q()
            self.ks = q2k*self.qs
            self.calculate_gamma(self.zeta)
            self.calculate_transfer_matrix(self.f, self.zeta)  
            
            # Calculate auxiliary matrices
            self.Ai_inv = exact_inv(self.Ai.copy())
            self.Ki_inv = exact_inv(self.Ki.copy()) # np.diag( np.exp(+1j*(2*np.pi*f*self.qs[:]*self.thick)/c_const) )
            self.Mrev = self.Ki[2:,2:]
            self.Mfwd = self.Ki_inv[0:2,0:2]
        
        # Fujita--Paren
        if self.modeltype == 'FP': 
            
            ai = self.ai_horiz
            bi_horiz = self.epsa + ai*(self.epsc-self.epsa)
            
            k1 = np.sqrt(bi_horiz[0]*(q2k)**2 + 1j*mu0*self.sigma*omega)
            k2 = np.sqrt(bi_horiz[1]*(q2k)**2 + 1j*mu0*self.sigma*omega)
            k1 = -np.conj(k1) # Waves are downward propagating (without this correction the phase coherences do not match the GTM model).
            k2 = -np.conj(k2)
            self.ks = np.array([k1,k2, k1,k2], dtype=np.complex128)
            self.qs = 1/q2k*self.ks
            self.Mrev = np.diag(np.exp(1j*self.ks[[2,3]]*self.thick))
            self.Mfwd = np.diag(np.exp(1j*self.ks[[0,1]]*self.thick))
            
            # Rotate principal frame back to the fabric frame
            self.Mrev = self.eig_to_meas_frame(self.Mrev)
            self.Mfwd = self.eig_to_meas_frame(self.Mfwd)


    def eig_to_meas_frame(self, mat_eig_frame):
        
        ang = -self.beta + self.eigang # Fabric eigen frame to measurement frame
        c, s = np.cos(ang), np.sin(ang)
        Qz = np.array(((c, -s), (s, c))) # rotation matrix
        return np.matmul(Qz, np.matmul(mat_eig_frame, Qz.T)) # = mat_meas_frame
       

#------------------------
# Fabric routines
#------------------------

def nlm_to_c2(nlm):
    
    nlm = np.divide(nlm, nlm[0]) # normalized coefficients [n_2^-2/n_0^0, n_2^-1/n_0^0, ..., n_2^2/n_0^0]
    n00, n22m,n21m,n20,n21p,n22p, *_ = nlm
    
    xi,yi,zi = 0,1,2 # indices
    c2ten = np.zeros((3,3), dtype=np.complex128)
    
    n20term = np.sqrt(2/3)*n20
    
    c2ten[xi,xi] = +np.real(n22p) - 1/2*n20term 
    c2ten[xi,yi] = -np.imag(n22p)
    c2ten[xi,zi] = -np.real(n21p)
    #
    c2ten[yi,xi] = c2ten[xi,yi] # sym
    c2ten[yi,yi] = -np.real(n22p) - 1/2*n20term
    c2ten[yi,zi] = +np.imag(n21p)
    #
    c2ten[zi,xi] = c2ten[xi,zi] # sym
    c2ten[zi,yi] = c2ten[yi,zi] # sym
    c2ten[zi,zi] = +n20term
    
    ccavg = np.eye(3)/3 + np.sqrt(2/15) * c2ten
    
    return np.real(ccavg)


def c2_to_nlm(c2, n00=1/np.sqrt(4*np.pi)):
    
    xi,yi,zi = 0,1,2 # indices
    
    M = (c2-np.eye(3)/3)/np.sqrt(2/15) # remove isotropic part and re-scale
    
    n20  = M[zi,zi]/np.sqrt(2/3)
    n21p = -M[xi,zi] + 1j*M[yi,zi]
    n22p = M[xi,xi]+M[zi,zi]/2 - 1j*M[xi,yi]

    nlm = n00*np.array([1, np.conj(n22p), -np.conj(n21p), n20, n21p, n22p])
    lm  = np.array([(0,0), (2,-2),(2,-1),(2,0),(2,1),(2,2)]).T 
    
    return (nlm, lm)


def eigenbasis(ccavg):
    
    ai, vi = lag.eig(ccavg)
    I = ai.argsort()[::-1]   
    eigvals = ai[I]
    e1, e2, e3 = vi[:,I[0]], vi[:,I[1]], vi[:,I[2]]
    
    return (eigvals, e1, e2, e3)