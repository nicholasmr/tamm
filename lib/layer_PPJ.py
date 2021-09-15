# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2021

"""
This is a shaved-down version of the "Layer" class by M. Jeannin (2019), implementing Passler and Paarmann (2017,2019):
https://github.com/pyMatJ/pyGTM/tree/master/GTM
"""

import numpy as np
import numpy.linalg as lag

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

c_const = 299792458 # m/s
eps0    = 8.8541878128e-12 ## vacuum permittivity
mu0     = 1.25663706212e-6 ## vaccum permeability
qsd_thr = 1e-10 ### threshold for wavevector comparison
    
    
def exact_inv(M):
    """Compute the 'exact' inverse of a 4x4 matrix using the analytical result. 
    
    Parameters
    ----------
    M : 4X4 array (float or complex)
      Matrix to be inverted
        
    Returns
    -------
    out : 4X4 array (complex)
        Inverse of this matrix or Moore-Penrose approximation if matrix cannot be inverted.
    Notes
    -----
    This should give a higher precision and speed at a reduced noise.
    From D.Dietze code https://github.com/ddietze/FSRStools
    
    .. seealso:: http://www.cg.info.hiroshima-cu.ac.jp/~miyazaki/knowledge/teche23.html
    
    """
    assert M.shape == (4, 4)

    # the following equations use algebraic indexing; transpose input matrix to get indexing right
    A = M.T
    detA = A[0, 0] * A[1, 1] * A[2, 2] * A[3, 3] + A[0, 0] * A[1, 2] * A[2, 3] * A[3, 1] + A[0, 0] * A[1, 3] * A[2, 1] * A[3, 2]
    detA = detA + A[0, 1] * A[1, 0] * A[2, 3] * A[3, 2] + A[0, 1] * A[1, 2] * A[2, 0] * A[3, 3] + A[0, 1] * A[1, 3] * A[2, 2] * A[3, 0]
    detA = detA + A[0, 2] * A[1, 0] * A[2, 1] * A[3, 3] + A[0, 2] * A[1, 1] * A[2, 3] * A[3, 0] + A[0, 2] * A[1, 3] * A[2, 0] * A[3, 1]
    detA = detA + A[0, 3] * A[1, 0] * A[2, 2] * A[3, 1] + A[0, 3] * A[1, 1] * A[2, 0] * A[3, 2] + A[0, 3] * A[1, 2] * A[2, 1] * A[3, 0]

    detA = detA - A[0, 0] * A[1, 1] * A[2, 3] * A[3, 2] - A[0, 0] * A[1, 2] * A[2, 1] * A[3, 3] - A[0, 0] * A[1, 3] * A[2, 2] * A[3, 1]
    detA = detA - A[0, 1] * A[1, 0] * A[2, 2] * A[3, 3] - A[0, 1] * A[1, 2] * A[2, 3] * A[3, 0] - A[0, 1] * A[1, 3] * A[2, 0] * A[3, 2]
    detA = detA - A[0, 2] * A[1, 0] * A[2, 3] * A[3, 1] - A[0, 2] * A[1, 1] * A[2, 0] * A[3, 3] - A[0, 2] * A[1, 3] * A[2, 1] * A[3, 0]
    detA = detA - A[0, 3] * A[1, 0] * A[2, 1] * A[3, 2] - A[0, 3] * A[1, 1] * A[2, 2] * A[3, 0] - A[0, 3] * A[1, 2] * A[2, 0] * A[3, 1]

    if detA == 0:
        return np.linalg.pinv(M)

    B = np.zeros(A.shape, dtype=np.complex128)
    B[0, 0] = A[1, 1] * A[2, 2] * A[3, 3] + A[1, 2] * A[2, 3] * A[3, 1] + A[1, 3] * A[2, 1] * A[3, 2] - A[1, 1] * A[2, 3] * A[3, 2] - A[1, 2] * A[2, 1] * A[3, 3] - A[1, 3] * A[2, 2] * A[3, 1]
    B[0, 1] = A[0, 1] * A[2, 3] * A[3, 2] + A[0, 2] * A[2, 1] * A[3, 3] + A[0, 3] * A[2, 2] * A[3, 1] - A[0, 1] * A[2, 2] * A[3, 3] - A[0, 2] * A[2, 3] * A[3, 1] - A[0, 3] * A[2, 1] * A[3, 2]
    B[0, 2] = A[0, 1] * A[1, 2] * A[3, 3] + A[0, 2] * A[1, 3] * A[3, 1] + A[0, 3] * A[1, 1] * A[3, 2] - A[0, 1] * A[1, 3] * A[3, 2] - A[0, 2] * A[1, 1] * A[3, 3] - A[0, 3] * A[1, 2] * A[3, 1]
    B[0, 3] = A[0, 1] * A[1, 3] * A[2, 2] + A[0, 2] * A[1, 1] * A[2, 3] + A[0, 3] * A[1, 2] * A[2, 1] - A[0, 1] * A[1, 2] * A[2, 3] - A[0, 2] * A[1, 3] * A[2, 1] - A[0, 3] * A[1, 1] * A[2, 2]

    B[1, 0] = A[1, 0] * A[2, 3] * A[3, 2] + A[1, 2] * A[2, 0] * A[3, 3] + A[1, 3] * A[2, 2] * A[3, 0] - A[1, 0] * A[2, 2] * A[3, 3] - A[1, 2] * A[2, 3] * A[3, 0] - A[1, 3] * A[2, 0] * A[3, 2]
    B[1, 1] = A[0, 0] * A[2, 2] * A[3, 3] + A[0, 2] * A[2, 3] * A[3, 0] + A[0, 3] * A[2, 0] * A[3, 2] - A[0, 0] * A[2, 3] * A[3, 2] - A[0, 2] * A[2, 0] * A[3, 3] - A[0, 3] * A[2, 2] * A[3, 0]
    B[1, 2] = A[0, 0] * A[1, 3] * A[3, 2] + A[0, 2] * A[1, 0] * A[3, 3] + A[0, 3] * A[1, 2] * A[3, 0] - A[0, 0] * A[1, 2] * A[3, 3] - A[0, 2] * A[1, 3] * A[3, 0] - A[0, 3] * A[1, 0] * A[3, 2]
    B[1, 3] = A[0, 0] * A[1, 2] * A[2, 3] + A[0, 2] * A[1, 3] * A[2, 0] + A[0, 3] * A[1, 0] * A[2, 2] - A[0, 0] * A[1, 3] * A[2, 2] - A[0, 2] * A[1, 0] * A[2, 3] - A[0, 3] * A[1, 2] * A[2, 0]

    B[2, 0] = A[1, 0] * A[2, 1] * A[3, 3] + A[1, 1] * A[2, 3] * A[3, 0] + A[1, 3] * A[2, 0] * A[3, 1] - A[1, 0] * A[2, 3] * A[3, 1] - A[1, 1] * A[2, 0] * A[3, 3] - A[1, 3] * A[2, 1] * A[3, 0]
    B[2, 1] = A[0, 0] * A[2, 3] * A[3, 1] + A[0, 1] * A[2, 0] * A[3, 3] + A[0, 3] * A[2, 1] * A[3, 0] - A[0, 0] * A[2, 1] * A[3, 3] - A[0, 1] * A[2, 3] * A[3, 0] - A[0, 3] * A[2, 0] * A[3, 1]
    B[2, 2] = A[0, 0] * A[1, 1] * A[3, 3] + A[0, 1] * A[1, 3] * A[3, 0] + A[0, 3] * A[1, 0] * A[3, 1] - A[0, 0] * A[1, 3] * A[3, 1] - A[0, 1] * A[1, 0] * A[3, 3] - A[0, 3] * A[1, 1] * A[3, 0]
    B[2, 3] = A[0, 0] * A[1, 3] * A[2, 1] + A[0, 1] * A[1, 0] * A[2, 3] + A[0, 3] * A[1, 1] * A[2, 0] - A[0, 0] * A[1, 1] * A[2, 3] - A[0, 1] * A[1, 3] * A[2, 0] - A[0, 3] * A[1, 0] * A[2, 1]

    B[3, 0] = A[1, 0] * A[2, 2] * A[3, 1] + A[1, 1] * A[2, 0] * A[3, 2] + A[1, 2] * A[2, 1] * A[3, 0] - A[1, 0] * A[2, 1] * A[3, 2] - A[1, 1] * A[2, 2] * A[3, 0] - A[1, 2] * A[2, 0] * A[3, 1]
    B[3, 1] = A[0, 0] * A[2, 1] * A[3, 2] + A[0, 1] * A[2, 2] * A[3, 0] + A[0, 2] * A[2, 0] * A[3, 1] - A[0, 0] * A[2, 2] * A[3, 1] - A[0, 1] * A[2, 0] * A[3, 2] - A[0, 2] * A[2, 1] * A[3, 0]
    B[3, 2] = A[0, 0] * A[1, 2] * A[3, 1] + A[0, 1] * A[1, 0] * A[3, 2] + A[0, 2] * A[1, 1] * A[3, 0] - A[0, 0] * A[1, 1] * A[3, 2] - A[0, 1] * A[1, 2] * A[3, 0] - A[0, 2] * A[1, 0] * A[3, 1]
    B[3, 3] = A[0, 0] * A[1, 1] * A[2, 2] + A[0, 1] * A[1, 2] * A[2, 0] + A[0, 2] * A[1, 0] * A[2, 1] - A[0, 0] * A[1, 2] * A[2, 1] - A[0, 1] * A[1, 0] * A[2, 2] - A[0, 2] * A[1, 1] * A[2, 0]

    out = B.T / detA
    return out


class Layer_PPJ:
    
    def __init__(self):
        
        self.M = np.zeros((6, 6), dtype=np.complex128) ## constitutive relations
        self.a = np.zeros((6, 6), dtype=np.complex128) ##
        self.S = np.zeros((4, 4), dtype=np.complex128) ##
        self.Delta = np.zeros((4, 4), dtype=np.complex128) ##
        self.qs = np.zeros(4, dtype=np.complex128) ## out of plane wavevector
        self.Py = np.zeros((3,4), dtype=np.complex128) ## Poyting vector
        self.gamma = np.zeros((4, 3), dtype=np.complex128) ##
        self.Ai = np.zeros((4, 4), dtype=np.complex128) ##
        self.Ki = np.zeros((4, 4), dtype=np.complex128) ##
        self.Ti = np.zeros((4, 4), dtype=np.complex128) ## Layer transfer matrix
        
    
    def calculate_matrices(self, zeta):
        """
        Calculate the principal matrices necessary for the GTM algorithm.
        
        Parameters
        ----------
        zeta : complex 
             In-plane reduced wavevector kx/k0 in the system. 
        
        Returns
        -------
             None
        
        Notes
        -----
        Note that zeta is conserved through the whole system and set externaly
        using the angle of incidence and `System.superstrate.epsilon[0,0]` value
        
        Requires prior execution of :py:func:`calculate_epsilon`
        
        """
        ## Constitutive matrix (see e.g. eqn (4))
        self.M[0:3, 0:3] = self.epsilon.copy()
        self.M[3:6, 3:6] = self.mu*np.identity(3)
        
        ## from eqn (10)
        b = self.M[2,2]*self.M[5,5] - self.M[2,5]*self.M[5,2]
        
        ## a matrix from eqn (9)
        self.a[2,0] = (self.M[5,0]*self.M[2,5] - self.M[2,0]*self.M[5,5])/b
        self.a[2,1] = ((self.M[5,1]-zeta)*self.M[2,5] - self.M[2,1]*self.M[5,5])/b
        self.a[2,3] = (self.M[5,3]*self.M[2,5] - self.M[2,3]*self.M[5,5])/b
        self.a[2,4] = (self.M[5,4]*self.M[2,5] - (self.M[2,4]+zeta)*self.M[5,5])/b
        self.a[5,0] = (self.M[5,2]*self.M[2,0] - self.M[2,2]*self.M[5,0])/b
        self.a[5,1] = (self.M[5,2]*self.M[2,1] - self.M[2,2]*(self.M[5,1]-zeta))/b
        self.a[5,3] = (self.M[5,2]*self.M[2,3] - self.M[2,2]*self.M[5,3])/b
        self.a[5,4] = (self.M[5,2]*(self.M[2,4]+zeta) - self.M[2,2]*self.M[5,4])/b
        
        ## S Matrix (Don't know where it comes from since Delta is just S re-ordered)
        ## Note that after this only Delta is used
        ### S[3,ii] was wrong in matlab code (M{kl}(4,6,:) should be M{kl}(5,6,:))
        self.S[0,0] = self.M[0,0] + self.M[0,2]*self.a[2,0] + self.M[0,5]*self.a[5,0];
        self.S[0,1] = self.M[0,1] + self.M[0,2]*self.a[2,1] + self.M[0,5]*self.a[5,1];
        self.S[0,2] = self.M[0,3] + self.M[0,2]*self.a[2,3] + self.M[0,5]*self.a[5,3];
        self.S[0,3] = self.M[0,4] + self.M[0,2]*self.a[2,4] + self.M[0,5]*self.a[5,4];
        self.S[1,0] = self.M[1,0] + self.M[1,2]*self.a[2,0] + (self.M[1,5]-zeta)*self.a[5,0];
        self.S[1,1] = self.M[1,1] + self.M[1,2]*self.a[2,1] + (self.M[1,5]-zeta)*self.a[5,1];
        self.S[1,2] = self.M[1,3] + self.M[1,2]*self.a[2,3] + (self.M[1,5]-zeta)*self.a[5,3];
        self.S[1,3] = self.M[1,4] + self.M[1,2]*self.a[2,4] + (self.M[1,5]-zeta)*self.a[5,4];
        self.S[2,0] = self.M[3,0] + self.M[3,2]*self.a[2,0] + self.M[3,5]*self.a[5,0];
        self.S[2,1] = self.M[3,1] + self.M[3,2]*self.a[2,1] + self.M[3,5]*self.a[5,1];
        self.S[2,2] = self.M[3,3] + self.M[3,2]*self.a[2,3] + self.M[3,5]*self.a[5,3];
        self.S[2,3] = self.M[3,4] + self.M[3,2]*self.a[2,4] + self.M[3,5]*self.a[5,4];
        self.S[3,0] = self.M[4,0] + (self.M[4,2]+zeta)*self.a[2,0] + self.M[4,5]*self.a[5,0];
        self.S[3,1] = self.M[4,1] + (self.M[4,2]+zeta)*self.a[2,1] + self.M[4,5]*self.a[5,1];
        self.S[3,2] = self.M[4,3] + (self.M[4,2]+zeta)*self.a[2,3] + self.M[4,5]*self.a[5,3];
        self.S[3,3] = self.M[4,4] + (self.M[4,2]+zeta)*self.a[2,4] + self.M[4,5]*self.a[5,4];
        
        
        ## Delta Matrix from eqn (8)
        self.Delta[0,0] = self.S[3,0]
        self.Delta[0,1] = self.S[3,3]
        self.Delta[0,2] = self.S[3,1]
        self.Delta[0,3] = - self.S[3,2]
        self.Delta[1,0] = self.S[0,0]
        self.Delta[1,1] = self.S[0,3]
        self.Delta[1,2] = self.S[0,1]
        self.Delta[1,3] = - self.S[0,2]
        self.Delta[2,0] = -self.S[2,0]
        self.Delta[2,1] = -self.S[2,3]
        self.Delta[2,2] = -self.S[2,1]
        self.Delta[2,3] = self.S[2,2]
        self.Delta[3,0] = self.S[1,0]
        self.Delta[3,1] = self.S[1,3]
        self.Delta[3,2] = self.S[1,1]
        self.Delta[3,3] = -self.S[1,2]      
        
       
    def calculate_q(self):
        """
        Calculates the 4 out-of-plane wavevectors for the current layer. 
        
        Returns
        -------
        None
            
        
        Notes
        -----
        From this we also get the Poynting vectors. 
        Wavevectors are sorted according to (trans-p, trans-s, refl-p, refl-s)
        Birefringence is determined according to a threshold value `qsd_thr` 
        set at the beginning of the script. 
        """
        Delta_loc = np.zeros((4,4), dtype=np.complex128)
        transmode = np.zeros((2), dtype=np.int)
        reflmode  = np.zeros((2), dtype=np.int)
        
        Delta_loc = self.Delta.copy()
        ## eigenvals // eigenvects as of eqn (11)
        qsunsorted, psiunsorted = lag.eig(Delta_loc)
        
        kt = 0 
        kr = 0;
        ## sort berremann qi's according to (12)
        if any(np.abs(np.imag(qsunsorted))):
            for km in range(0,4):
                if np.imag(qsunsorted[km])>=0 :
                    transmode[kt] = km
                    kt = kt + 1
                else:
                    reflmode[kr] = km
                    kr = kr +1
        else:
            for km in range(0,4):
                if np.real(qsunsorted[km])>0 :
                    transmode[kt] = km
                    kt = kt + 1
                else:
                    reflmode[kr] = km
                    kr = kr +1
        ## Calculate the Poyting vector for each Psi using (16-18)
        for km in range(0,4):
            Ex = psiunsorted[0,km] 
            Ey = psiunsorted[2,km]
            Hx = -psiunsorted[3,km]
            Hy = psiunsorted[1,km]
            ## from eqn (17)
            Ez = self.a[2,0]*Ex + self.a[2,1]*Ey + self.a[2,3]*Hx + self.a[2,4]*Hy
            # from eqn (18)
            Hz = self.a[5,0]*Ex + self.a[5,1]*Ey + self.a[5,3]*Hx + self.a[5,4]*Hy
            ## and from (16)
            self.Py[0,km] = Ey*Hz-Ez*Hy
            self.Py[1,km] = Ez*Hx-Ex*Hz
            self.Py[2,km] = Ex*Hy-Ey*Hx
            
        ## check Cp using either the Poynting vector for birefringent
        ## materials or the electric field vector for non-birefringent
        ## media to sort the modes       
        
        ## first calculate Cp for transmitted waves
        Cp_t1 = np.abs(self.Py[0,transmode[0]])**2/(np.abs(self.Py[0,transmode[0]])**2+np.abs(self.Py[1,transmode[0]])**2)
        Cp_t2 = np.abs(self.Py[0,transmode[1]])**2/(np.abs(self.Py[0,transmode[1]])**2+np.abs(self.Py[1,transmode[1]])**2)
        
        if np.abs(Cp_t1-Cp_t2) > qsd_thr: ## birefringence
            if Cp_t2>Cp_t1:
                transmode = np.flip(transmode,0) ## flip the two values
            ## then calculate for reflected waves if necessary
            Cp_r1 = np.abs(self.Py[0,reflmode[1]])**2/(np.abs(self.Py[0,reflmode[1]])**2+np.abs(self.Py[1,reflmode[1]])**2)
            Cp_r2 = np.abs(self.Py[0,reflmode[0]])**2/(np.abs(self.Py[0,reflmode[0]])**2+np.abs(self.Py[1,reflmode[0]])**2)
            if Cp_r1>Cp_r2:
                reflmode = np.flip(reflmode,0) ## flip the two values
        
        else:     ### No birefringence, use the Electric field s-pol/p-pol
            Cp_te1 = np.abs(psiunsorted[0,transmode[1]])**2/(np.abs(psiunsorted[0,transmode[1]])**2+np.abs(psiunsorted[2,transmode[1]])**2)
            Cp_te2 = np.abs(psiunsorted[0,transmode[0]])**2/(np.abs(psiunsorted[0,transmode[0]])**2+np.abs(psiunsorted[2,transmode[0]])**2)
            if Cp_te1>Cp_te2:
                transmode = np.flip(transmode,0) ## flip the two values
            Cp_re1 = np.abs(psiunsorted[0,reflmode[1]])**2/(np.abs(psiunsorted[0,reflmode[1]])**2+np.abs(psiunsorted[2,reflmode[1]])**2)  
            Cp_re2 = np.abs(psiunsorted[0,reflmode[0]])**2/(np.abs(psiunsorted[0,reflmode[0]])**2+np.abs(psiunsorted[2,reflmode[0]])**2)  
            if Cp_re1>Cp_re2:
                reflmode = np.flip(reflmode,0) ## flip the two values
        
        ## finaly store the sorted version        
        ####### q is (trans-p, trans-s, refl-p, refl-s)
        self.qs[0] = qsunsorted[transmode[0]]
        self.qs[1] = qsunsorted[transmode[1]]
        self.qs[2] = qsunsorted[reflmode[0]]
        self.qs[3] = qsunsorted[reflmode[1]]
        Py_temp = self.Py.copy()      
        self.Py[:,0] = Py_temp[:,transmode[0]]
        self.Py[:,1] = Py_temp[:,transmode[1]]
        self.Py[:,2] = Py_temp[:,reflmode[0]]
        self.Py[:,3] = Py_temp[:,reflmode[1]]
        
    def calculate_gamma(self, zeta):
        """
        Calculate the gamma matrix
        
        Parameters
        ----------
        zeta : complex
             in-plane reduced wavevector kx/k0
        
        Returns
        -------
        None
        """
        ### this whole function is eqn (20)
        self.gamma[0,0] = 1.0 + 0.0j
        self.gamma[1,1] = 1.0 + 0.0j
        self.gamma[3,1] = 1.0 + 0.0j
        self.gamma[2,0] = -1.0 + 0.0j
        
        if np.abs(self.qs[0]-self.qs[1])<qsd_thr:
            gamma12 = 0.0 + 0.0j
            
            gamma13 = -(self.mu*self.epsilon[2,0]+zeta*self.qs[0])
            gamma13 = gamma13/(self.mu*self.epsilon[2,2]-zeta**2)
            
            gamma21 = 0.0 + 0.0j
            
            gamma23 = -self.mu*self.epsilon[2,1]
            gamma23 = gamma23/(self.mu*self.epsilon[2,2]-zeta**2)
        
        else:
            gamma12_num = self.mu*self.epsilon[1,2]*(self.mu*self.epsilon[2,0]+zeta*self.qs[0])
            gamma12_num = gamma12_num - self.mu*self.epsilon[1,0]*(self.mu*self.epsilon[2,2]-zeta**2)
            gamma12_denom = (self.mu*self.epsilon[2,2]-zeta**2)*(self.mu*self.epsilon[1,1]-zeta**2-self.qs[0]**2)
            gamma12_denom = gamma12_denom - self.mu**2*self.epsilon[1,2]*self.epsilon[2,1]
            gamma12 = gamma12_num/gamma12_denom
            if np.isnan(gamma12):
                gamma12 = 0.0 + 0.0j
            
            gamma13 = -(self.mu*self.epsilon[2,0]+zeta*self.qs[0])
            gamma13 = gamma13-self.mu*self.epsilon[2,1]*gamma12 #### gamma12 factor missing in (20) but present in ref [13]
            gamma13 = gamma13/(self.mu*self.epsilon[2,2]-zeta**2)
            
            if np.isnan(gamma13):
                gamma13 = -(self.mu*self.epsilon[2,0]+zeta*self.qs[0])
                gamma13 = gamma13/(self.mu*self.epsilon[2,2]-zeta**2)

            gamma21_num = self.mu*self.epsilon[2,1]*(self.mu*self.epsilon[0,2]+zeta*self.qs[1])
            gamma21_num = gamma21_num-self.mu*self.epsilon[0,1]*(self.mu*self.epsilon[2,2]-zeta**2)
            gamma21_denom = (self.mu*self.epsilon[2,2]-zeta**2)*(self.mu*self.epsilon[0,0]-self.qs[1]**2)
            gamma21_denom = gamma21_denom-(self.mu*self.epsilon[0,2]+zeta*self.qs[1])*(self.mu*self.epsilon[2,0]+zeta*self.qs[1])
            gamma21 = gamma21_num/gamma21_denom
            if np.isnan(gamma21):
                gamma21 = 0.0+0.0j
                
            gamma23 = -(self.mu*self.epsilon[2,0] +zeta*self.qs[1])*gamma21-self.mu*self.epsilon[2,1]
            gamma23 = gamma23/(self.mu*self.epsilon[2,2]-zeta**2)
            if np.isnan(gamma23):
                gamma23 = -self.mu*self.epsilon[2,1]/(self.mu*self.epsilon[2,2]-zeta**2)

        if np.abs(self.qs[2]-self.qs[3])<qsd_thr:
            gamma32 = 0.0 + 0.0j
            gamma33 = (self.mu*self.epsilon[2,0]+zeta*self.qs[2])/(self.mu*self.epsilon[2,2]-zeta**2)
            gamma41 = 0.0 + 0.0j
            gamma43 = -self.mu*self.epsilon[2,1]/(self.mu*self.epsilon[2,2]-zeta**2)
        
        else:
            gamma32_num = self.mu*self.epsilon[1,0]*(self.mu*self.epsilon[2,2]+zeta**2)
            gamma32_num = gamma32_num-self.mu*self.epsilon[1,2]*(self.mu*self.epsilon[2,0]+zeta*self.qs[2])
            gamma32_denom = (self.mu*self.epsilon[2,2]-zeta**2)*(self.mu*self.epsilon[1,1]-zeta**2-self.qs[2]**2)
            gamma32_denom = gamma32_denom-self.mu**2*self.epsilon[1,2]*self.epsilon[2,1]
            gamma32 = gamma32_num/gamma32_denom
            if np.isnan(gamma32):
                gamma32 = 0.0 + 0.0j
            
            gamma33 = self.mu*self.epsilon[2,0] + zeta*self.qs[2]
            gamma33 = gamma33 + self.mu*self.epsilon[2,1]*gamma32 #### gamma32 factor missing in (20) but present in ref [13]
            gamma33 = gamma33/(self.mu*self.epsilon[2,2]-zeta**2)
            if np.isnan(gamma33):
                gamma33 = (self.mu*self.epsilon[2,0] + zeta*self.qs[2])/(self.mu*self.epsilon[2,2]-zeta**2)
            
            gamma41_num = self.mu*self.epsilon[2,1]*(self.mu*self.epsilon[0,2]+zeta*self.qs[3])
            gamma41_num = gamma41_num - self.mu*self.epsilon[0,1]*(self.mu*self.epsilon[2,2]-zeta**2)
            gamma41_denom = (self.mu*self.epsilon[2,2]-zeta**2)*(self.mu*self.epsilon[0,0]-self.qs[3]**2)
            gamma41_denom = gamma41_denom - (self.mu*self.epsilon[0,2]+zeta*self.qs[3])*(self.mu*self.epsilon[2,0]+zeta*self.qs[3])
            gamma41 = gamma41_num/gamma41_denom
            if np.isnan(gamma41):
                gamma41 = 0.0 + 0.0j
                
            gamma43 = -(self.mu*self.epsilon[2,0]+zeta*self.qs[3])*gamma41
            gamma43 = gamma43-self.mu*self.epsilon[2,1]
            gamma43 = gamma43/(self.mu*self.epsilon[2,2]-zeta**2)
            if np.isnan(gamma43):
                gamma43 = -self.mu*self.epsilon[2,1]/(self.mu*self.epsilon[2,2]-zeta**2)
        
        ### gamma field vectors should be normalized to avoid any birefringence problems
        # use double square bracket notation to ensure correct array shape
        gamma1 = np.array([[self.gamma[0,0], gamma12, gamma13]],dtype=np.complex128)
        gamma2 = np.array([[gamma21, self.gamma[1,1], gamma23]],dtype=np.complex128)
        gamma3 = np.array([[self.gamma[2,0], gamma32, gamma33]],dtype=np.complex128)
        gamma4 = np.array([[gamma41, self.gamma[3,1], gamma43]],dtype=np.complex128)
        gamma1 = gamma1/np.sqrt(np.matmul(gamma1,gamma1.T)) # normalize
        gamma2 = gamma2/np.sqrt(np.matmul(gamma2,gamma2.T)) # normalize
        gamma3 = gamma3/np.sqrt(np.matmul(gamma3,gamma3.T)) # normalize
        gamma4 = gamma4/np.sqrt(np.matmul(gamma4,gamma4.T)) # normalize
        
        self.gamma[0,:] = gamma1
        self.gamma[1,:] = gamma2
        self.gamma[2,:] = gamma3
        self.gamma[3,:] = gamma4
        
        
    def calculate_transfer_matrix(self, f, zeta):
        """
        Compute the transfer matrix of the whole layer :math:`T_i=A_iP_iA_i^{-1}`
        
        Parameters
        ----------
        f : float 
            frequency (in Hz)
        zeta : complex
               reduced in-plane wavevector kx/k0
        Returns
        -------
        None
        
        """
        ## eqn(22)
        self.Ai[0,:] = self.gamma[:,0].copy()
        self.Ai[1,:] = self.gamma[:,1].copy()
        
        self.Ai[2,:] = (self.qs*self.gamma[:,0]-zeta*self.gamma[:,2])/self.mu
        self.Ai[3,:] = self.qs*self.gamma[:,1]/self.mu
        
        for ii in range(4):
            ## looks a lot like eqn (25). Why is K not Pi ?
            self.Ki[ii,ii] = np.exp(-1.0j*(2.0*np.pi*f*self.qs[ii]*self.thick)/c_const)
        
        # OUT COMMENTED
        #Aim1 = exact_inv(self.Ai.copy())
        ### eqn (26)
        #self.Ti = np.matmul(self.Ai,np.matmul(self.Ki,Aim1))
        