#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2021

import copy, sys, code # code.interact(local=locals())
import numpy as np

from scipy import interpolate
import scipy.special as sp

sys.path.insert(0, '../../lib')

from plottools import *
from specfabpy import specfabpy as sf # requires the spectral fabric module to be compiled!

# For plotting parcel geometry
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import patches

s2yr   = 3.16887646e-8
yr2s   = 31556926    
yr2kyr = 1e-3 

class PureShear():

    def __init__(self, t_c, r, ax='z'): 

        # e-folding time scale: 
        # if t_c > 0 ==> half-height time
        # if t_c < 0 ==> double-height time
        self.t_c = float(t_c)/np.log(2) 

        if ax=='z': self.Fpow = [(1+r)/2., (1-r)/2., -1]
        if ax=='y': self.Fpow = [(1+r)/2., -1, (1-r)/2.]
        if ax=='x': self.Fpow = [-1, (1+r)/2., (1-r)/2.]

    def lam(self, t):    return np.exp(t/self.t_c) # lambda(t)
    def F(self, t):      return np.diag(np.power(self.lam(t),self.Fpow))
    
    def strain(self, t): return 0.5*( self.F(t) + np.transpose(self.F(t)) ) - np.diag([1,1,1])
    #def epszz2time(self,epszz): return -self.t_e*np.log(epszz+1) # time it takes to reach "eps_zz" strain with t_e char. timescale

    # Note that F is constructed such that W and eps are time-independant.
    def D(self): return 1/self.t_c * np.diag(self.Fpow)
    def W(self): return np.diag([0,0,0])

class SimpleShear(): 

    def __init__(self, t_c, plane='zx'): 

        # Shear strain rate
        self.kappa = 1/t_c # t_c is the time taken to reach a shear strain of 1

        # Shear plane
        if plane=='zx': self.Frel = np.outer([1,0,0],[0,0,1])
        if plane=='yx': self.Frel = np.outer([1,0,0],[0,1,0])

    def F(self, t): return np.eye(3) + np.tan(self.time2beta(t))*self.Frel
    
    def time2beta(self, time): return np.arctan(self.kappa*time)
    def beta2time(self, beta): return np.tan(beta)/self.kappa
    
    # Note that F is constructed such that W and eps are time-independant.
    def ugrad(self): return self.kappa*self.Frel
    def D(self): return 0.5 * (self.ugrad() + self.ugrad().T) 
    def W(self): return 0.5 * (self.ugrad() - self.ugrad().T) 


#-------------------
# SynthFab
#-------------------

class SyntheticFabric():
    
    def __init__(self, H=2000, N=100, L=10, nu=2.5e-3): 
        
        self.H  = H # ice mass height
        self.N  = N # number of layers        
        self.L  = L # spectral truncation 
        self.nu0 = nu # Regularization strength (calibrated for L and strain-rate)
               
        self.nlm_len = sf.init(self.L) 
        self.lm = sf.get_lm(self.nlm_len)
        
        self.nlm = np.zeros((self.N, self.nlm_len), dtype=np.complex128) 
        self.nlm[0,0] = 1/np.sqrt(4*np.pi) # init with isotropy
   
    def intp_nlm(self, nlm_list, depth):
        
        nlm_list_intp = np.zeros((self.N,self.nlm_len), dtype=np.complex128) 
        z_intp = np.linspace(0, depth[-1], self.N)
        
        for lmii in np.arange(self.nlm_len):
        
            intp = interpolate.interp1d(depth, nlm_list[:,lmii])
            nlm_list_intp[:,lmii] = intp(z_intp)
        
        return (nlm_list_intp, -z_intp)
    
    def plot_side(self, ax, x,y,z, alpha=0, scale=1, lw=1.25, color='k'):
        verts = scale*np.array([list(zip(x,y,z))])
        coll = Poly3DCollection(verts)
        coll.set_edgecolor('0.4')
        coll.set_facecolor(color)
        coll.set_linewidth(lw)
        coll.set_alpha(alpha)
        coll.set_clip_on(False)
        ax.add_collection3d(coll)
    
    def plot_parcel(self, ax, xyz0, dzx,dzy,dyx, color='k', plotaxlbls=False):
               
        x0,y0,z0   = xyz0 
        ax.view_init(20, +70+180)
        
        x,y,z = [0,dyx,x0+dyx,x0], [0,y0,y0,0], [0,0,0,0] # bottom
        self.plot_side(ax, x,y,z, alpha=0.4, color=color)

        x,y,z = [0,dyx,dzx+dyx,dzx], [0,y0,y0+dzy,dzy], [0,0,z0,z0] # left
        self.plot_side(ax, x,y,z, alpha=0.3, color=color)
        
        x,y,z = [0,x0,x0+dzx,dzx,dzx], [0,0,dzy,dzy], [0,0,z0,z0] # back
        self.plot_side(ax, x,y,z, alpha=0.3, color=color)
        
        x,y,z = [dzx,dzx+dyx,x0+dzx+dyx,x0+dzx], [dzy,y0+dzy,y0+dzy,dzy], [z0,z0,z0,z0] # top
        self.plot_side(ax, x,y,z, alpha=0.1, color=color)
        
        x,y,z = [dyx,x0+dyx,x0+dzx+dyx,dzx+dyx], [y0,y0,y0+dzy,y0+dzy], [0,0,z0,z0] # front
        self.plot_side(ax, x,y,z, alpha=0.3, color=color)
        
        x,y,z = [x0,x0+dyx,x0+dzx+dyx,x0+dzx], [0,y0,y0+dzy,dzy], [0,0,z0,z0] # right
        self.plot_side(ax, x,y,z, alpha=0.3, color=color)
        
        #ax.scatter([0],[0],[0], 'o', s=[50], color='k')
        lw=1.5
        zero, one = np.array([0,0]), np.array([0,1])
        xspan, yspan, zspan = one, one, one
        ax.plot(xspan,zero,zero, '-', lw=lw, color='k', zorder=10)
        ax.plot(zero,yspan,zero, '-', lw=lw, color='k', zorder=10)
        ax.plot(zero,zero,zspan, '-', lw=lw, color='k', zorder=10)
        if plotaxlbls:
            axlenmul = 1.15
            ax.text(xspan.max()*axlenmul , 0, 0, r"$\vu{x}$", color='k', zorder=10)
            ax.text(-xspan.max()*0.2, yspan.max()*axlenmul, 0, r"$\vu{y}$", color='k', zorder=10)
            ax.text(0, 0, zspan.max()*axlenmul , r"$\vu{z}$", color='k', zorder=10)               
        
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); 
        ax.set_xticks(np.linspace(0,1,5),minor=True)
        ax.set_yticks(np.linspace(0,1,5),minor=True)
        ax.set_zticks(np.linspace(0,1,5),minor=True)
        ax.set_xlim(xspan); ax.set_ylim(yspan); ax.set_zlim(zspan)
        
        ax.set_axis_off()
        
       
        return
    
    # This is slightly different from that in "plottools.py" --- it is adapted for the synthfabric plot window.
    def plot_ODF(self, ax, nlm, lm, geo=None, rot0=0, cmap='Greys', tickintvl=2, cblbl=r'$\psi(\theta,\phi)/N$', title=''):
        
        F, lon,lat = discretize_ODF(nlm, lm)
        lvls = np.linspace(0.1,0.5,5) # Contour lvls
        hdistr = ax.contourf(np.rad2deg(lon), np.rad2deg(lat), F, transform=ccrs.PlateCarree(), levels=lvls, extend='both', cmap=cmap)
        kwargs_gridlines = {'ylocs':np.arange(-90,90+30,30), 'xlocs':np.arange(0,360+45,45), 'linewidth':0.5, 'color':'black', 'alpha':0.25, 'linestyle':'-'}
        gl = ax.gridlines(crs=ccrs.PlateCarree(), **kwargs_gridlines)
        gl.xlocator = mticker.FixedLocator(np.array([-135, -90, -45, 0, 90, 45, 135, 180]))
        cb1 = plt.colorbar(hdistr, ax=ax, fraction=0.065, aspect=10,  orientation='vertical', pad=0.1, ticks=lvls[::tickintvl])   
        cb1.set_label(cblbl)
        cb1.ax.yaxis.set_ticks(lvls, minor=True)
        ax.set_title(title, fontsize=FS)
        
        if geo is not None:
            ax.plot([0],[90],'k.', transform=geo)
            #ax.plot([0],[0],'kx', transform=geo)
            ax.plot([rot0],[0],'k.', transform=geo)
            ax.text(rot0-80, 70, r'$\vu{z}$', horizontalalignment='left', transform=geo)
            ax.text(rot0-18, -8, r'$\vu{y}$', horizontalalignment='left', transform=geo)
        
        return hdistr
    
    def makeProfile(self, deformExpr1, deformExpr2, dt=10, t_end=1000, crossover=[0.5,0.7], plot=True):
    
        dt    *= yr2s
        t_end *= yr2s
        
        Nt = int(t_end/dt)# Total number of integration steps taken
        t0 = int(Nt*crossover[0]) 
        t1 = int(Nt*crossover[1])
        tii = [t0,Nt-t1] # number of steps per deformation regime
        
        ### Construct strain-rate and spin tensor histories
        
        D = np.zeros((2,3,3)) # strain-rate
        W = np.zeros((2,3,3)) # spin
        
        # Parcel deformation for plotting
        xyz0_init = (1,1,1)
        xyz0 = [xyz0_init,xyz0_init]
        dzx, dzy, dyx = [0,0], [0,0], [0,0]
        
        # Determine D and W for fabric evolution
        for ii, deformExpr in enumerate((deformExpr1,deformExpr2)):
            
            if 'ax' in deformExpr:
                PS = PureShear(deformExpr['t_c']*yr2s, deformExpr['r'], ax=deformExpr['ax'])
                W[ii,:,:], D[ii,:,:] = PS.W(), PS.D()
                xyz0[ii] = np.matmul(PS.F(tii[ii]*dt), np.array(xyz0[ii]))
                #print(xyz0[ii])
                        
            elif 'plane' in deformExpr: 
                SS = SimpleShear(deformExpr['t_c']*yr2s, plane=deformExpr['plane']) 
                W[ii,:,:], D[ii,:,:] = SS.W(), SS.D()
                xyz0[ii] = xyz0_init
                ex,ey,ez = np.array([xyz0_init[0],0,0]),np.array([0,xyz0_init[1],0]),np.array([0,0,xyz0_init[2]])
                dyx[ii] = np.dot(np.matmul(SS.F(tii[ii]*dt)-np.eye(3), ey), ex)
                dzx[ii] = np.dot(np.matmul(SS.F(tii[ii]*dt)-np.eye(3), ez), ex)
                dzy[ii] = 0
                #print(dzx[ii])

            else:
                sys.exit('Invalid deformation type')

        ### Fabric evolution 
        
        nlm_list      = np.zeros((Nt+1,self.nlm_len), dtype=np.complex128) # The expansion coefficients
        nlm_list[0,0] = self.nlm[0,0] # Normalized such that N(t=0) = 1
        
        for tt in np.arange(1,Nt+1):
            
            if tt <= t0: g = 1
            if tt >= t1: g = 0
            if t0 < tt < t1: g = (t1-tt)/(t1-t0)
           
            D_ = g*D[0,:,:] + (1-g)*D[1,:,:]
            W_ = g*W[0,:,:] + (1-g)*W[1,:,:]
            
            # Regularization
            dndt = sf.nu(self.nu0, D_) * sf.dndt_reg(nlm_list[tt-1,:]) 
           
            # Lattice rotation
            dndt += sf.dndt_latrot(nlm_list[tt-1,:], D_,W_) 
            
            # DDRX
            tau = D_ # *** assumes stress and strain-rate are co-axial *** (magnitude does not matter because decay rate is normalized)
            Gamma0 = g*deformExpr1['Gamma0'] + (1-g)*deformExpr2['Gamma0']
            dndt += Gamma0 * sf.dndt_ddrx(nlm_list[tt-1,:], tau) 
            
            nlm_list[tt,:] = nlm_list[tt-1,:] + dt * np.matmul(dndt, nlm_list[tt-1,:])
        
        depth = np.linspace(0, self.H, Nt+1)
        (self.nlm, z_intp) = self.intp_nlm(nlm_list, depth)
        self.z = z_intp
        
        ### Plot
        
        if plot:

            scale = 0.4
            fig = plt.figure(figsize=(15*scale,11*scale))
            
            gs_master = gridspec.GridSpec(1, 3, width_ratios=[1.1,0.6,0.45])
            gs_master.update(top=0.94, bottom=0.12, left=-0.14, right=1-0.1, hspace=0.3, wspace=0.2)
            gs_parcels = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs_master[0], hspace=0.1, height_ratios=[1,1,0.25,1,1])
            gs         = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_master[2], hspace=0.45)
 
            axEig = fig.add_subplot(gs_master[:, 1])
            #
            axParcel = [fig.add_subplot(gs_parcels[ii, 0], projection='3d') for ii in (0,1,3,4)]
            #
            inclination = 45 # view angle
            rot0 = 1 * -90
            rot = 1*-20 + rot0 #*1.4 # view angle
            prj = ccrs.Orthographic(rot, 90-inclination)
            geo = ccrs.Geodetic()            
            axODF = [fig.add_subplot(gs[ii, 0], projection=prj) for ii in np.arange(3)]
            axODF[0].set_global()
            axODF[1].set_global()
            axODF[2].set_global()
            
            ##
            
            N0 = int(t0/Nt * (self.N-1))
            N1 = int(t1/Nt * (self.N-1))
            
            ai = np.zeros((self.N,3))
            for ii in np.arange(self.N): ai[ii,:] = sf.frame(self.nlm[ii,:], 'e')[3]

            # plot colors (colorbrewer)
            cb = '#1f78b4'
            cr = '#e31a1c'
            cg = '#33a02c'
            
            # lighter colors
            crl = '#fb9a99'
            cbl = '#a6cee3'

            lw = 1.5
            zkm = 1e-3*z_intp
            axEig.plot(ai[:,0],zkm,'-k',  lw=lw, label=r'$\lambda_1$')
            axEig.plot(ai[:,1],zkm,'--k', lw=lw, label=r'$\lambda_2$')
            axEig.plot(ai[:,2],zkm,':k',  lw=lw, label=r'$\lambda_3$')
            #
            axEig.plot([0,1],[zkm[N0],zkm[N0]],'--', color=cr, lw=lw)
            axEig.plot([0,1],[zkm[N1],zkm[N1]],'--', color=cg, lw=lw)
            axEig.plot([0,1],[zkm[-1],zkm[-1]],'--', color=cb, lw=lw, clip_on=False)
            #
            rect1 = patches.Rectangle((0,zkm[N0]), 1, -zkm[N0], color='#fee5d9')
            rect2 = patches.Rectangle((0,zkm[N1]), 1, -(zkm[N1]-zkm[N0]), color='#edf8e9')
            rect3 = patches.Rectangle((0,zkm[-1]), 1, -(zkm[-1]-zkm[N1]), color='#deebf7')
            axEig.add_patch(rect1); axEig.add_patch(rect2); axEig.add_patch(rect3)
            #
            axEig.set_xticks(np.arange(0,1+1e-3, 0.5))
            axEig.set_xticks(np.arange(0,1+1e-3, 0.10), minor=True)
            axEig.set_xlim([0,1])
            axEig.set_yticks(np.arange(zkm[-1], 0+1e-6, 0.50))
            axEig.set_yticks(np.arange(zkm[-1], 0+1e-6, 0.1), minor=True)
            axEig.set_ylim(zkm[[-1,0]]*1.01)
            axEig.set_xlabel('$\lambda_i$')
            axEig.set_ylabel('$z$ (km)')
            #
            legkwargs = {'frameon':True, 'fancybox':False, 'edgecolor':'k', 'framealpha':0.9, 'ncol':1, 'handlelength':1.34, 'labelspacing':0.3}
            hleg = axEig.legend(loc=1, **legkwargs)
            hleg.get_frame().set_linewidth(0.7);
            
            ###
            
            self.plot_parcel(axParcel[0], xyz0_init, 0,0,0, color=crl, plotaxlbls=True)
            self.plot_parcel(axParcel[2], xyz0_init, 0,0,0, color=cbl, plotaxlbls=True)
            self.plot_parcel(axParcel[1], xyz0[0], dzx[0],dzy[0],dyx[0], color=cr)
            self.plot_parcel(axParcel[3], xyz0[1], dzx[1],dzy[1],dyx[1], color=cb)

            self.plot_ODF(axODF[0], self.nlm[N0,:], self.lm, cmap='Reds',   geo=geo, rot0=rot0, title=r'$z=\SI{%1.1f}{\kilo\metre}$'%(zkm[N0]))        
            self.plot_ODF(axODF[1], self.nlm[N1,:], self.lm, cmap='Greens', geo=geo, rot0=rot0, title=r'$z=\SI{%1.1f}{\kilo\metre}$'%(zkm[N1]))        
            self.plot_ODF(axODF[2], self.nlm[-1,:], self.lm, cmap='Blues',  geo=geo, rot0=rot0, title=r'$z=\SI{%1.1f}{\kilo\metre}$'%(zkm[-1]))        
        
        if 0: 
            fout = 'test.png'
            print('Saving %s'%(fout))
            plt.savefig(fout,dpi=300)

        plt.show()           
    
        return (self.nlm, self.lm, self.z)
    
### DEBUG

if 0:    
    synfab = SyntheticFabric(N=100)
    
    #t_e = 1/((3/yr2s)/synfab.H) # eps_zz = (mean accum)/H
    t_c = 1000
    deformExpr1 = {'ax':'z', 't_c':+t_c, 'r':0, 'Gamma0':0}
    deformExpr2 = {'ax':'z', 't_c':+t_c, 'r':0, 'Gamma0':0e-9}
    
    #kappa = 1e-10 # shear strain-rate 
    #kappa = 2e-3
    t_c = 1000
    deformExpr1 = {'plane':'zx', 't_c':t_c, 'Gamma0':0e-9}
    
    synfab.makeProfile(deformExpr1, deformExpr2, t_end=1000, crossover=[1,1])    
    
       
