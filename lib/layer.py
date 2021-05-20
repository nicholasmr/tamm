# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2021

"""
This subclass wraps a spectral fabric formulation around the Passler and Paarmann (2017,2019) layer class by M. Jeannin (2019)
"""

import numpy as np
import numpy.linalg as lag

from layer_GTM import *

class Layer(Layer_GTM):
    
    def __init__(self, n2m, thickness, f, zeta, epsa, epsc, sigma, mu): 
        
        super().__init__()
        
        self.thick  = thickness # layer thickness (d)
        self.f      = f # plane wave frequency
        self.zeta   = zeta 
        #
        self.epsa  = epsa # permitivity perdendicular to c-axis
        self.epsc  = epsc # permitivity parallel to c-axis
        self.sigma = sigma # single crystal isotropic conductivity
        self.mu    = mu # relative permeability
        
        self.init_with_n2m(n2m)
        
    def init_with_n2m(self, n2m):
        
        self.n2m = n2m
        self.set_fabric(self.n2m)
        self.epsilon = self.get_eps_from_n2m(self.n2m)  
        
        self.init_with_eps(self.epsilon)
        
    def init_with_eps(self, eps):
        
        self.epsilon = eps ## epsilon is a 3x3 matrix of permittivity at a given frequency
        
        # Call parent GTM routines
        self.calculate_matrices(self.zeta)
        self.calculate_q()
        self.calculate_gamma(self.zeta)
        self.calculate_transfer_matrix(self.f, self.zeta)  
        
        # Calculate auxiliary matrices
        self.Ai_inv = exact_inv(self.Ai.copy())
        self.Ki_inv = exact_inv(self.Ki.copy()) # np.diag( np.exp(+1j*(2*np.pi*f*self.qs[:]*self.thick)/c_const) )
        self.Mrev = self.Ki[2:,2:]
        self.Mfwd = self.Ki_inv[0:2,0:2]

        
    def set_fabric(self, n2m):
        
        self.n2m = n2m # normalized ODF coef vector [n22m/n00, n21m/n00, n20/n00, n21p/n00, n22p/n00]
        nlm = np.hstack(([1],self.n2m))
        self.ccavg = self.get_c2_from_n2m(n2m) # <c outer c>

    def get_eps_from_n2m(self, n2m):
        
        monopole   = (2*self.epsa+self.epsc)/3 * np.eye(3)
        quadrupole = (self.epsc-self.epsa) * (self.ccavg - np.eye(3)/3)
        epsavg     = np.real(monopole+quadrupole) # Should be real. If fabric is rotated using e.g. a complex phase, numerical errors might cause small but non-zero imaginary values.
        
        epsdprime = self.sigma/(2*np.pi*self.f*eps0)
        epsavg = epsavg - 1j*epsdprime * np.eye(3)
        
        return epsavg

    def get_c2_from_n2m(self, n2m):
        
        n22m,n21m,n20,n21p,n22p = n2m
        xi,yi,zi = 0,1,2 # indices
        c2ten = np.zeros((3,3), dtype=np.complex)
        
        n20term = np.sqrt(2/3)*n20
        #
        c2ten[xi,xi] = +(n22p+n22m) - n20term 
        c2ten[xi,yi] = -1j*(n22m-n22p)
        c2ten[xi,zi] =    +(n21m-n21p)
        #
        c2ten[yi,xi] = c2ten[xi,yi] ### sym
        c2ten[yi,yi] = -(n22p+n22m) - n20term
        c2ten[yi,zi] = -1j*(n21m+n21p)
        #
        c2ten[zi,xi] = c2ten[xi,zi] ### sym
        c2ten[zi,yi] = c2ten[yi,zi] ### sym
        c2ten[zi,zi] = +2*n20term
        
        return np.eye(3)/3 + 1/np.sqrt(30) * c2ten
    
    def get_eigframe(self):
        
        eigvals, eigvecs = lag.eig(self.ccavg)
        I = eigvals.argsort()[::-1]   
        self.eigvals = eigvals[I]
        self.e1, self.e2, self.e3 = eigvecs[:,I[0]], eigvecs[:,I[1]], eigvecs[:,I[2]]
        
        return (self.eigvals, self.e1, self.e2, self.e3)
