import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from scipy.stats.distributions import uniform
from forward_mapping import *
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14

fmap = forward_mapping()
nsample = 100

a=np.zeros((nsample,fmap.nt+1))
b=np.zeros((nsample,fmap.nt+1))

pl.figure(figsize=(12,8))
for index in np.arange(nsample):
    pars = fmap.diff_coeff_exact(fmap.tpars)[1:]
    std = np.mean(pars)*10.0**-1
    noise = np.random.randn(fmap.npars)
    pars += std*noise
    pars = np.log(pars)
    pars = np.insert(pars,fmap.npars,1.0*10.0**2*np.random.uniform())
    theta = np.insert(pars,0,np.log(fmap.pars0))
    my_fsc = fmap.fscub(np.exp(theta[:-1]),0)
    my_k = my_fsc(fmap.t)
    my_soln = fmap.solve(my_k)

    a[index,:] = my_soln[0,:]
    b[index,:] = my_soln[1,:]
    pl.subplot(3,1,1)    
    pl.plot(fmap.t,fmap.ro*fmap.cp*my_k,'k',lw=0.1,alpha=1.0)
    #plt.xticks([])
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
pl.plot(fmap.t,fmap.ro*fmap.cp*my_k,'k',lw=0.1,alpha=1.0,label=r'$Thermal\;conductivity\;coefficient$')
pl.grid()
pl.legend(loc=0)
pl.subplot(3,1,2)
# pl.plot(fmap.t,a[index,:],'k')
pl.semilogy(fmap.t,np.var(a,axis=0),'k',label=r'$Temperature\;variance\;at\;r=0$')
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
pl.grid()
pl.legend(loc=0)
pl.subplot(3,1,3)
# pl.plot(fmap.t,b[index,:],'k')
pl.semilogy(fmap.t,np.var(b,axis=0),'k',label=r'$Variance\;of\;T\;at\;r=R_0$')
pl.legend(loc=0)
pl.grid()
pl.xlabel('Time (Seconds)')
pl.savefig('uncertainty_propagation.png')
