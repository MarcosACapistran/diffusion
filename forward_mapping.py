#coding: utf8

from scipy import interpolate as ip
import pickle
import numpy as np
import scipy as sp
import pylab as pl
import time
import os
"""
Este programa crea una temperatura sintética resolviendo con un k 
propuesto y la perturba mediante un error gaussiano la relación de señal a ruido
es SNR=(max(u)/std(u))=100. Se trabaja con dos propuestas de k. Una cóncava y una 
sigmoide. Para elegir entre ellas hay que descomentar/comentar las líneas adecuadas: 
descomentar 87/91 para la cóncava (al revés para la sigmoide).
"""

# set the state of random
#np.random.seed(seed=100)

class forward_mapping:
    """
    Esta clase genera datos sintéticos con mallado fino y grueso
    """
    def __init__(self,npars=50,R=0.045,T=1000.0,T0=295.0,ro=1000.6,cp = 3780.0,
    alfa=4.217*10**(-4),beta=(120.0/61.0)*10**6,n=100,dilat=100,nt=350,pars0 = 1.1897623330763447*10**-7):
        self.R = R
        self.T = T # Tiempo final
        self.T0 = T0
        self.ro = ro
        self.cp = cp
        self.alfa = alfa
        self.beta = beta
        self.coef = self.alfa*self.beta/(self.ro*self.cp)
        self.H = 40.0/(self.ro*self.cp)
        self.C = self.R**2/4.0
        self.dilat = dilat
        self.pars0 = pars0
        # Mallado espacial
        self.n = n # Número de puntos interiores de la malla espacial
        self.h = self.R/(self.n+1) # Paso en espacio
        self.r = np.linspace(0.0,self.R,self.n+2) # Malla espacial
        self.onesn2 = np.ones(self.n+2)
        self.u0 = self.T0*self.onesn2
        self.m = 1.0/(2.0*np.arange(1,self.n+2))
        self.mn2 = np.insert(1.0-self.m[:self.n],self.n,2.0)
        self.m02 = np.insert(1.0+self.m[:self.n],0,2.0)
        # Malla parámetros
        self.npars = npars # Número de parámetros
        self.taupars = self.T/self.npars # Paso de tiempo en la malla de parámetros
        self.tpars = np.linspace(0.0,self.T,self.npars+1) # Malla parámetros
        # Malla gruesa
        self.nt = nt # Número de intervalos en la malla gruesa
        self.tau = self.T/self.nt # Paso de tiempo en la malla gruesa
        self.t = np.linspace(0.0,self.T,self.nt+1) # Malla gruesa
        self.aux1 = self.tau*self.H*(1.0+1.0/(2.0*self.n+2.0))/self.h
        self.aux2 = 2.0*self.aux1*self.T0
        self.ct = self.coef*self.tau/2.0
        self.fi = self.coef/(np.exp(self.coef*self.t[1:])-1.0)
        # Malla fina
        self.ntx = self.dilat*self.nt # Número de intervalos en la malla fina
        self.taux = self.T/self.ntx # Paso de tiempo en la malla fina
        self.tx = np.linspace(0.0,self.T,self.ntx+1) # Malla fina
        self.aux1x = self.taux*self.H*(1.0+1.0/(2.0*self.n+2.0))/self.h
        self.aux2x = 2.0*self.aux1x*self.T0
        self.ctx = self.coef*self.taux/2.0
        # Restricciones
        self.tmed=(self.t[:-1]+self.t[1:])/2.0
        self.fimed = self.coef/(np.exp(self.coef*self.tmed)-1.0)
        self.aconstr=-self.tau*self.fimed/3.0-np.ones(self.nt)
        self.bconstr=np.zeros(self.nt)
        self.bconstr[:-1]=-self.tau*(self.fimed[:-1]+4*self.fi[:-1]+self.fimed[1:])/3.0
        self.bconstr[-1]=-self.tau*(self.fimed[-1]+2*self.fi[-1])/3.0+1.0
        self.cconstr=-self.tau*self.fimed/3.0+np.ones(self.nt)
        self.smconstr=np.zeros(self.nt)
        self.smconstr[0]=-self.aconstr[0]*self.pars0    
    # Funciones auxiliares del problema inverso
    def multridiag(self,a,b,c,y): # (I-B)*u
        return (self.onesn2-b)*y-np.insert(a*y[:-1],0,0.0)-np.insert(c*y[1:],len(c),0.0)
    def tridiag(self,a,b,c,d):
        nn = len(b)
        m = np.zeros(nn)
        g = np.zeros(nn)
        v = np.zeros(nn)
        m[0] = b[0]
        g[0] = d[0]/m[0]
        for i in np.arange(1,nn):
            m[i]=b[i]-(c[i-1]/m[i-1])*a[i-1]
            g[i]=(d[i]-g[i-1]*a[i-1])/m[i]
        v[-1] = g[-1]
        for i in np.arange(nn-2,-1,-1):
            v[i] = g[i]-(c[i]/m[i])*v[i+1]
        return v    
    # k exacta
    def diff_coeff_exact(self,t):
        """
        k cóncava
        """
        return (np.arctan(t/30.0)+0.45)/(self.ro*self.cp)
        """
        k sigmoide
        """
        #return ((np.arctan((t-450.0)/150.0)+3.5)/2.0)/(self.ro*self.cp)
    # Splines cúbicos en los parámetros
    def fscub(self,theta,s):
        # s is either 0 or 1 to interpolate or adjust respectively
        if s == 0:
            return ip.UnivariateSpline(self.tpars,theta,k=1,s=0)
            #return ip.CubicSpline(self.tpars,theta,bc_type='clamped')
        if s == 1:
            return ip.UnivariateSpline(self.tpars,theta,k=1,s=0)
            #return ip.CubicSpline(self.tpars,theta,bc_type='clamped')        
    # Solución en la malla gruesa
    def solve(self,k):
        s = k*self.tau/(2*self.h**2)
        u = np.zeros((self.n+2,self.nt+1))
        u[:,0] = self.u0
        a = -s[0]*self.mn2
        b = (2.0*s[0]-self.ct)*self.onesn2
        c = -s[0]*self.m02
        b[-1] += self.aux1
        for j in np.arange(self.nt):
            d = self.multridiag(a,b,c,u[:,j])
            d[-1] += self.aux2
            a = -s[j+1]*self.mn2
            b = (2.0*s[j+1]-self.ct)*self.onesn2
            b[-1] += self.aux1
            c = -s[j+1]*self.m02
            u[:,j+1] = self.tridiag(a,self.onesn2+b,c,d)
        return np.array([u[0,:],u[-1,:]])
    # Solución en la malla fina
    def solvexac(self,k):
        s = k*self.taux/(2*self.h**2)
        u = np.zeros((self.n+2,self.ntx+1))
        u[:,0] = self.u0
        a = -s[0]*self.mn2
        b = (2.0*s[0]-self.ctx)*self.onesn2
        c = -s[0]*self.m02
        b[-1] += self.aux1x
        for j in np.arange(self.ntx):
            d = self.multridiag(a,b,c,u[:,j])
            d[-1] += self.aux2x
            a = -s[j+1]*self.mn2
            b = (2.0*s[j+1]-self.ctx)*self.onesn2
            b[-1] += self.aux1x
            c = -s[j+1]*self.m02
            u[:,j+1] = self.tridiag(a,self.onesn2+b,c,d)
        return np.array([u[0,:],u[-1,:]])
    # Genera datos sintéticos usando la malla fina
    def create_data(self):
        k = self.diff_coeff_exact(self.tx)
        u = self.solvexac(k)
        u = u[:,np.arange(0,self.ntx+1,self.dilat)] # Solución sintética
        self.std = 1.0*u.mean()/10.0**3
        v = u + self.std*np.random.randn(2,self.nt+1) # Mediciones
        v[:,0]=[self.T0,self.T0] # Los valores iniciales quedan sin error
        return u,v

    # Genera datos sintéticos usando la malla gruesa
    def create_data_coarse(self):
        # k = self.diff_coeff_exact(self.tx)
        # u = self.solvexac(k)
        # u = u[:,np.arange(0,self.ntx+1,self.dilat)] # Solución sintética
        k = self.diff_coeff_exact(self.t)
        u = self.solve(k) # Solución sintética        
        self.std = 1.0*u.mean()/10.0**3
        v = u + self.std*np.random.randn(2,self.nt+1) # Mediciones
        v[:,0]=[self.T0,self.T0] # Los valores iniciales quedan sin error
        return u,v
