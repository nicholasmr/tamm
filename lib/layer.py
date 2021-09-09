# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2021

"""
This subclass wraps a spectral fabric formulation around the Passler and Paarmann (2017,2019) layer class implemented by M. Jeannin (2019)
"""

import numpy as np
import numpy.linalg as lag

from layer_GTM import *

class Layer(Layer_GTM):
    
    def __init__(self, n2m, beta, thickness, f, zeta, epsa, epsc, sigma, mu, modeltype='GTM'): 
        
        super().__init__()
        
        self.modeltype = modeltype # GTM (General Transfer Matrix, aka. tamm) or FP (Fujita--Paren)
        
        self.thick  = thickness # layer thickness (d)
        self.f      = f         # plane wave frequency
        self.zeta   = zeta      # x-component of wave vector (dimensionless)
        
        self.epsa  = epsa  # permitivity perdendicular to c-axis
        self.epsc  = epsc  # permitivity parallel to c-axis
        self.sigma = sigma # single crystal isotropic conductivity
        self.mu    = mu    # relative permeability
        
        self.init_with_n2m(n2m, beta)
        
        
    def init_with_n2m(self, n2m, beta):
        
        self.set_fabric(n2m, beta)
        self.epsilon = self.get_eps_from_n2m(self.n2m)  
        self.init_with_eps(self.epsilon)
        

    def set_fabric(self, n2m, beta):
    
        # Fabric frame 
        self.n2m_0   = n2m
        self.ccavg_0 = self.get_c2_from_n2m(n2m) 
        
        # Measurement frame
        self.beta  = beta
        self.n2m   = self.rotate_frame(n2m, beta) # normalized ODF coef vector [n22m/n00, n21m/n00, n20/n00, n21p/n00, n22p/n00]
        self.ccavg = self.get_c2_from_n2m(self.n2m) # <c \outer c>
        
        
    def rotate_frame(self, n2m, beta):
        
        rotmat = np.diag([np.exp(+1j*m*beta) for m in np.arange(-2,2+1)])
        return np.matmul(rotmat,n2m)
    
    
    def init_with_eps(self, eps):
        
        self.epsilon = eps # 3x3 matrix of permittivity at a given frequency
        
        # Horizontal eigenvalues 
        self.ai_horiz, self.Qeig = lag.eig(self.ccavg_0[:2,:2])
        self.Qeiginv = self.Qeig.T # lag.inv(self.Qeig)
        
        self.eigang = np.arctan2(self.Qeig[1,0],self.Qeig[0,0])
        
        # Short hands
        omega = 2*np.pi*self.f
        q2k = omega/c_const        
        
        # Setup layer matrices
        if self.modeltype == 'GTM': # General transfer matrix
        
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
        
        if self.modeltype == 'FP': # Fujita--Paren
            
            ai = self.ai_horiz
            #ai *= (self.ccavg_0[0,0] + self.ccavg_0[1,1]) / (ai[0]+ai[1])
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


    def eig_to_meas_frame(self, diagmat):
        
        # Eigenframe to fabric frame
        X_fabfrm = diagmat
        
        # Fabric frame to measurement frame
        b = -self.beta 
        b += self.eigang
        
        c, s = np.cos(b), np.sin(b)
        Qz = np.array(((c, -s), (s, c))) # rotation matrix
        X_measfrm = np.matmul(Qz, np.matmul(X_fabfrm, Qz.T))        
        
        return X_measfrm
        

    def get_eps_from_n2m(self, n2m):
        
        monopole   = (2*self.epsa+self.epsc)/3 * np.eye(3)
        quadrupole = (self.epsc-self.epsa) * (self.ccavg - np.eye(3)/3)
        epsavg     = np.real(monopole+quadrupole) # Should be real. If fabric is rotated using e.g. a complex phase, numerical errors might cause small but non-zero imaginary values.
        
        epsdprime = self.sigma/(2*np.pi*self.f*eps0)
        epsavg = epsavg - 1j*epsdprime * np.eye(3)
        
        return epsavg


    @staticmethod
    def get_c2_from_n2m(n2m):
        
        n22m,n21m,n20,n21p,n22p = n2m
        xi,yi,zi = 0,1,2 # indices
        c2ten = np.zeros((3,3), dtype=np.complex128)
        
        n20term = np.sqrt(2/3)*n20
        
#        c2ten[xi,xi] = +(n22p+n22m) - n20term 
#        c2ten[xi,yi] = -1j*(n22m-n22p)
#        c2ten[xi,zi] =    +(n21m-n21p)
#        #
#        c2ten[yi,xi] = c2ten[xi,yi] ### sym
#        c2ten[yi,yi] = -(n22p+n22m) - n20term
#        c2ten[yi,zi] = -1j*(n21m+n21p)
#        #
#        c2ten[zi,xi] = c2ten[xi,zi] ### sym
#        c2ten[zi,yi] = c2ten[yi,zi] ### sym
#        c2ten[zi,zi] = +2*n20term
#
#        ccavg = np.eye(3)/3 + 1/np.sqrt(30) * c2ten
        
        c2ten[xi,xi] = +np.real(n22p) - 1/2*n20term 
        c2ten[xi,yi] = -np.imag(n22p)
        c2ten[xi,zi] = -np.real(n21p)
        #
        c2ten[yi,xi] = c2ten[xi,yi] ### sym
        c2ten[yi,yi] = -np.real(n22p) - 1/2*n20term
        c2ten[yi,zi] = +np.imag(n21p)
        #
        c2ten[zi,xi] = c2ten[xi,zi] ### sym
        c2ten[zi,yi] = c2ten[yi,zi] ### sym
        c2ten[zi,zi] = +n20term
        
        ccavg = np.eye(3)/3 + np.sqrt(2/15) * c2ten
        
        return np.real(ccavg)
    
    
    @staticmethod
    def get_n2m_from_c2(c2, n00=1/np.sqrt(4*np.pi)):
        
        xi,yi,zi = 0,1,2 # indices
        
        M = (c2-np.eye(3)/3)/np.sqrt(2/15) # remove isotropic part
        
        n20  = M[zi,zi]/np.sqrt(2/3)
        n21p = -M[xi,zi] + 1j*M[yi,zi]
        n22p = M[xi,xi]+M[zi,zi]/2 - 1j*M[xi,yi]
        
        return n00*np.array([np.conj(n22p), -np.conj(n21p), n20, n21p, n22p])
    
    
    def get_eigframe(self):
        
        ai, vi = lag.eig(self.ccavg)
        I = ai.argsort()[::-1]   
        self.eigvals = ai[I]
        self.e1, self.e2, self.e3 = vi[:,I[0]], vi[:,I[1]], vi[:,I[2]]
        
        return (self.eigvals, self.e1, self.e2, self.e3)
