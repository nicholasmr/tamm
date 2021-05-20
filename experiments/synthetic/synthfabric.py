#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2021

import copy, sys, code # code.interact(local=locals())
import numpy as np
from scipy import interpolate

from specfabpy import specfabpy as sf # requires the spectral fabric module to be compiled!

s2yr   = 3.16887646e-8
yr2s   = 31556926    
yr2kyr = 1e-3 

class SynthFabric():
    
    def __init__(self, N, L=12, dL=4, H=2000): 
        
        self.H  = H # ice mass height
        
        self.N  = N # no. of layers        
        self.L  = L # spectral truncation 
        #self.dL = dL # L->L+dL for non-vertical fabrics for heigher zonal resolution
        
        self.nlm_len = sf.init(self.L) 
        
        self.nu0 = 5.0e-2 # Regularization strength (calibrated for L)
        
        # Some generic stress states used for warm DRX regimes
        mat = np.array([[0,0,1], [0,0,0], [0,0,0]])
        self.tau_xz = (mat+np.transpose(mat))/2
        mat = np.array([[0,1,0], [0,0,0], [0,0,0]])
        self.tau_xy = (mat+np.transpose(mat))/2
        
    def generate_all(self):
        
        #self.save_profile("dome",        self.VerticalCompression(epszz_end=-0.95, r=0) )
        
        self.save_profile("ridge",       self.VerticalCompression(epszz_end=-0.95, r=+1) )
        self.save_profile("icestream",   self.IceStream() )
        self.save_profile("shearmargin", self.ShearMargin() )
        
    def save_profile(self, exprname, fabtuple):
        
        (z, nlm, lm) = fabtuple        
        fname = "synthprofile_%s.h5"%(exprname)
        print('*** Saving %s'%(fname))
        with open(fname, 'wb') as f:
            np.save(f, nlm)
            np.save(f, lm)
            np.save(f, z)
            np.save(f, self.N)
        f.close()
            
    def load_profile(self, exprname):

        fname = "synthprofile_%s.h5"%(exprname)
        with open(fname, 'rb') as f:
            nlm = np.load(f)
            lm  = np.load(f)
            z   = np.load(f)
            N   = np.load(f)
        f.close()
        
        return (z,nlm,lm,N)
    
    
    def intp_nlm(self, nlm_list, depth):
        
        nlm_list_intp = np.zeros((self.N,self.nlm_len), dtype=np.complex128) 
        z_intp = np.linspace(0, depth[-1], self.N)
        
        for lmii in np.arange(self.nlm_len):
        
            intp = interpolate.interp1d(depth, nlm_list[:,lmii])
            nlm_list_intp[:,lmii] = intp(z_intp)
        
        return (nlm_list_intp, z_intp)
    
    #-------------------
    # Ridge like
    #-------------------
    
    def VerticalCompression(self, epszz_end=-0.95, r=0):

        ### Large-scale deformation
        
        # Constant strain-rate params
        b = 0.5 / yr2s # m ice equiv / second
        T = 1/(b/self.H) # eps_zz = (mean accum)/H
        
        deformobj = PureShear(T, r, vertdir='z') # single-maximum along z
        omg, eps = deformobj.W(), deformobj.eps()
                
        tend = deformobj.epszz2time(epszz_end)
        dt   = deformobj.epszz2time(0.5*-20e-3)
        Nt   = int(np.ceil(tend/dt))
        time = np.arange(Nt+1)*float(dt) # +1 to include the init state for t=0
        
        epszz = np.array([deformobj.strain(t)[-1,-1] for ii, t in enumerate(time)])
        depth = self.H*(epszz+1) - self.H
        
        ### Fabric evolution 
        
        nlm_list = np.zeros((Nt+1,self.nlm_len), dtype=np.complex128) # The expansion coefficients
        nlm_list[0,0] = 1/np.sqrt(4*np.pi) # Normalized such that N(t=0) = 1

        # Regularization
        dndt = sf.get_nu_eps(self.nu0, eps)*sf.get_dndt_ij_reg(self.nlm_len) 
        
        # Lattice rotation
        dndt += sf.get_dndt_ij_rot(self.nlm_len, eps,omg, eps*0,0,1,1,1) 
        
        for tt in np.arange(1,Nt+1):
            nlm_list[tt,:] = nlm_list[tt-1,:] + dt*np.matmul(dndt, nlm_list[tt-1,:])
        
        (nlm_list_intp, z_intp) = self.intp_nlm(nlm_list, depth)
        return (z_intp, nlm_list_intp, sf.get_lm(self.nlm_len))
    
    #-------------------
    # Ice stream like
    #-------------------
    
    def IceStream(self):
        
        BOT_DRX_OR_ROT = 1 # 0 = DRX dominant at bottom 1 = lattice rotation dominant at bottom
        
        ### Large-scale deformation
        
        dt    = 1*yr2s
        Nt    = 1000
        depth = np.linspace(0,self.H,Nt+1)
        
        b = 1.0 * 5.5 / yr2s # m ice equiv / second
        T = 1/(b/self.H) # eps_zz = (mean accum)/H
        deformobj = PureShear(-T, 0, vertdir='x') 
        omg_up, eps_up = deformobj.W(), deformobj.eps()

        kappa_dn = 7*1.5e-10 # shear strain-rate 
        deformobjs = SimpleShear(kappa_dn, shearplane='xz') 
        omg_dn, eps_dn = deformobjs.W(), deformobjs.eps()

        ### Fabric evolution 
                
        nlm_list = np.zeros((Nt+1,self.nlm_len), dtype=np.complex128) # The expansion coefficients
        nlm_list[0,0] = 1/np.sqrt(4*np.pi) # Normalized such that N(t=0) = 1
        
        t0 = int(Nt*7/10) 
        t1 = int(Nt*10/10)
        for tt in np.arange(1,Nt+1):
            
            if tt <= t0: w = 1
            if tt >= t1: w = 0
            if t0 <= tt <= t1: w = (t1-tt)/(t1-t0)
            
            # Regularization
            dndt = sf.get_nu_eps(self.nu0, eps_up)*sf.get_dndt_ij_reg(self.nlm_len) 
           
            # Lattice rotation
            eps = w*eps_up + (1-w)*eps_dn*BOT_DRX_OR_ROT
            omg = w*omg_up + (1-w)*omg_dn*BOT_DRX_OR_ROT
            
            dndt += sf.get_dndt_ij_rot(self.nlm_len, eps,omg, eps*0,0,1,1,1) 
            
            # DRX
            Gamma0 = 8e-10 * (1-w) * (1-BOT_DRX_OR_ROT)
            dndt += Gamma0*sf.get_dndt_ij_drx(self.nlm_len, nlm_list[tt-1,:], self.tau_xz) 
            
            nlm_list[tt,:] = nlm_list[tt-1,:] + dt*np.matmul(dndt, nlm_list[tt-1,:])
        
        (nlm_list_intp, z_intp) = self.intp_nlm(nlm_list, depth)
        return (z_intp, nlm_list_intp, sf.get_lm(self.nlm_len))
    
    #-------------------
    # Shear margin like
    #-------------------
        
    def ShearMargin(self):
        
        ### Large-scale deformation
        
        dt    = 1*yr2s
        Nt    = 1000
        depth = np.linspace(0,self.H,Nt+1)
        
        kappa_up = 1.5e-10 # shear strain-rate 
        deformobj = SimpleShear(-kappa_up, shearplane='xy') 
        omgp, epsp = deformobj.W(), deformobj.eps()

        kappa_dn = kappa_up # shear strain-rate 
        deformobjs = SimpleShear(kappa_dn, shearplane='xz') 
        omgs, epss = deformobjs.W(), deformobjs.eps()

        ### Fabric evolution 
        
        nlm_list = np.zeros((Nt+1,self.nlm_len), dtype=np.complex128) # The expansion coefficients
        nlm_list[0,0] = 1/np.sqrt(4*np.pi) # Normalized such that N(t=0) = 1
        
        t0 = int(Nt*7/10) 
        t1 = int(Nt*10/10)
        for tt in np.arange(1,Nt+1):
            
            if tt <= t0: w = 1
            if tt >= t1: w = 0
            if t0 <= tt <= t1: w = (t1-tt)/(t1-t0)
            
            # Regularization
            dndt = sf.get_nu_eps(self.nu0, epsp)*sf.get_dndt_ij_reg(self.nlm_len) 
            
            # Lattice rotation
            eps = w*epsp + 1*(1-w)*epss
            omg = w*omgp + 1*(1-w)*omgs
            dndt += sf.get_dndt_ij_rot(self.nlm_len, eps,omg, eps*0,0,1,1,1) 
            
            # DRX
            Gamma0 = 1e-10
            Gamma0_up = 8e0*Gamma0*w
            Gamma0_dn = 3e1*Gamma0*(1-w)
            dndt += Gamma0_up*sf.get_dndt_ij_drx(self.nlm_len, nlm_list[tt-1,:], self.tau_xy) 
            dndt += Gamma0_dn*sf.get_dndt_ij_drx(self.nlm_len, nlm_list[tt-1,:], self.tau_xz) 
            
            nlm_list[tt,:] = nlm_list[tt-1,:] + dt*np.tensordot(dndt, nlm_list[tt-1,:], axes=(-1,0))
        
        (nlm_list_intp, z_intp) = self.intp_nlm(nlm_list, depth)
        return (z_intp, nlm_list_intp, sf.get_lm(self.nlm_len))
    
   
#-------------------
# PURE SHEAR (PS)
#-------------------

class PureShear():

    def __init__(self, t_e, r, vertdir='z'): 

        self.t_e = float(t_e) # e-folding time scale

        if vertdir=='z': self.Fpow = [(1+r)/2., (1-r)/2., -1]
        if vertdir=='y': self.Fpow = [(1+r)/2., -1, (1-r)/2.]
        if vertdir=='x': self.Fpow = [-1, (1+r)/2., (1-r)/2.]

    def lam(self, t):    return np.exp(t/self.t_e) # lambda(t)
    def F(self, t):      return np.diag(np.power(self.lam(t),self.Fpow))
    def strain(self, t): return 0.5*( self.F(t) + np.transpose(self.F(t)) ) - np.diag([1,1,1])

    def epszz2time(self,epszz): return -self.t_e*np.log(epszz+1) # time it takes to reach "eps_zz" strain with t_e char. timescale

    # Note that F is constructed such that W and eps are time-independant.
    def W(self):   return np.diag([0,0,0])
    def eps(self): return 1./self.t_e * np.diag(self.Fpow)


#-------------------
# SIMPLE SHEAR (SS) 
#-------------------

class SimpleShear():

    def __init__(self, kappa, shearplane='xz'): 

        self.kappa0 = kappa    
        self.kappa = [0,0]

        if shearplane=='xz': self.kappa[0] = self.kappa0
        if shearplane=='xy': self.kappa[1] = self.kappa0

    def time2shearangle(self, t): return np.arctan(self.kappa0*t)
    
    def shearangle2time(self, ang): return np.tan(ang)/self.kappa0
    
    # Note that F is constructed such that W and eps are time-independant.
    def W(self):   return 0.5*np.matrix([[0,+self.kappa[1],+self.kappa[0]], [-self.kappa[1],0,0], [-self.kappa[0],0,0]]);
    def eps(self): return 0.5*np.matrix([[0,+self.kappa[1],+self.kappa[0]], [+self.kappa[1],0,0], [+self.kappa[0],0,0]]);

#-------------------
# RIGID ROTATION (RR) 
#-------------------

class RigidRotation():

    def __init__(self, omega, rotax='z'): 

        self.omega0 = omega
        self.omega = [0,0,0]

        if rotax=='x': self.omega[0] = self.omega0
        if rotax=='y': self.omega[1] = self.omega0
        if rotax=='z': self.omega[2] = self.omega0

    def beta(self, t): return self.omega0*t

    # Note that F is constructed such that W and eps are time-independant.
    def W(self):   return np.matrix([[0,-self.omega[2],self.omega[1]], [self.omega[2],0,-self.omega[0]], [-self.omega[1],self.omega[0],0]]);
    def eps(self): return np.diag([0,0,0])

