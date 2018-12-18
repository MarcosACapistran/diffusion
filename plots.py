#coding: utf8

from scipy import interpolate as ip
import numpy as np
import pylab as pl
import corner
import sys
import pickle
from forward_mapping import *
import matplotlib
#matplotlib.style.use('bmh')
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14

'''
Dibuja el map y la media del caminante, sus interpoladas y 
las temperaturas halladas con ellas. También dibuja las marginales
'''

x2 = np.int(sys.argv[1]) # indice desde donde se alcanza el equilibrio en el MCMC

fmap = forward_mapping()
#fmap = forward_mapping(pars0 = 1.1725473550631119e-06)
"""
Leemos lo que necesitamos del marco guardado del problema inverso
"""
settings = pickle.load(open("inv_prob_setting.pk","rb"))
ndim = settings.get('ndim')
u_exac = settings.get('u')
u_noise = settings.get('v')
# Encuentra el número del MAP
f = open("output0/probs.dat", "r")
a = f.read()
b = a.replace('[','')
c = b.replace(']','')
d = c.replace('inf','0.0 ')
a = d.split()
a = list(map(float,a))
a = np.asarray(a)
a = -a[x2:]
f.close()
nmap = np.int(np.where(a==a.max())[0][0])
prob_map = a[nmap]
# Grafica el MAP
f = open("output0/chain.dat",'r')
b = f.read()
c = b.replace('[','')
d = c.replace(']','')
b = d.split()
b = list(map(float,b))
chain = np.asarray(b)
chain = np.reshape(chain,(np.int(np.size(chain)/ndim),ndim))
chain = chain[x2:,:]
chain[:,:-1] = np.exp(chain[:,:-1])
f.close()
# Media, MAP y k exacta en la malla gruesa
muestra_mean = np.mean(chain[:,:-1],axis=0)
muestra_map = chain[nmap,:-1]
muestra_mean = np.insert(muestra_mean,0,fmap.pars0)
muestra_map = np.insert(muestra_map,0,fmap.pars0)
k_exact = fmap.diff_coeff_exact(fmap.tpars)
# Comparación de la verdad, el MAP y la media
pl.figure(figsize=(12,10))
pl.subplot(211)
# pl.plot(fmap.tpars,k_exact,'b-',label=r'$k_{True}$')
# pl.plot(fmap.tpars,muestra_map,'k-o',label=r'$k_{MAP}$')
# pl.plot(fmap.tpars,muestra_mean,'r-*',label=r'$k_{CM}$')
pl.plot(fmap.tpars,fmap.ro*fmap.cp*k_exact,'b-',label=r'$k_{True}$')
pl.plot(fmap.tpars,fmap.ro*fmap.cp*muestra_map,'k-o',label=r'$k_{MAP}$')
pl.plot(fmap.tpars,fmap.ro*fmap.cp*muestra_mean,'r-*',label=r'$k_{CM}$')
for zzz in np.arange(100):
    muestra_last = chain[-zzz,:-1]
    muestra_last = np.insert(muestra_last,0,fmap.pars0)
    pl.plot(fmap.tpars,fmap.ro*fmap.cp*muestra_last,'g-',lw=0.1,alpha=0.05)
muestra_last = chain[-1,:-1]
muestra_last = np.insert(muestra_last,0,fmap.pars0)
pl.plot(fmap.tpars,fmap.ro*fmap.cp*muestra_last,'g-',label=r'$K_{sample}$',lw=0.1,alpha=0.05)
pl.legend(loc=0)
pl.ylabel(r'$k(t)$')
pl.grid()

# pl.subplot(221)
# pl.plot(fmap.tpars,k_exact,'b-',label=r'$k_{True}$')
# pl.plot(fmap.tpars,muestra_map,'k-o',label=r'$k_{MAP}$')
# pl.plot(fmap.tpars,muestra_mean,'r-*',label=r'$k_{CM}$')
# for zzz in np.arange(100):
#     muestra_last = chain[-zzz,:-1]
#     muestra_last = np.insert(muestra_last,0,fmap.pars0)
#     pl.plot(fmap.tpars,muestra_map,'g-',alpha=0.5)
# muestra_last = chain[-1,:-1]
# muestra_last = np.insert(muestra_last,0,fmap.pars0)
# pl.plot(fmap.tpars,muestra_map,'g-',label=r'$K_{sample}$',alpha=0.5)
# pl.legend(loc=0)
# pl.ylabel(r'$T(t)$')
# pl.grid()
# # k exacta e interpoladas de la media y del MAP
k_exact_fina = fmap.diff_coeff_exact(fmap.t)
muestra_mean_ip = fmap.fscub(muestra_mean,1)(fmap.t)
muestra_map_ip = fmap.fscub(muestra_map,1)(fmap.t)
# #Comparación de la verdad, el MAP y la media interpolados
# pl.subplot(222)
# pl.plot(fmap.t,muestra_map_ip,'k-')
# pl.plot(fmap.t,muestra_mean_ip,'r-')
# pl.plot(fmap.t,k_exact_fina,'b-')
# pl.grid()
# # Temperaturas (del MAP, de la media, mediciones y exacta) en
# # el centro de la bola y en la frontera.
u_map = fmap.solve(muestra_map_ip)
u_mean = fmap.solve(muestra_mean_ip)
# Temperaturas (del MAP, de la media, mediciones y exacta) en el centro de la bola
pl.subplot(223)
pl.plot(fmap.t,u_exac[0,:]-u_map[0,:],'k-',label='map')
pl.plot(fmap.t,u_exac[0,:]-u_mean[0,:],'r-',label='mean')
#pl.plot(fmap.t,u_exac[0,:],'b-',label='exact')
pl.plot(fmap.t,u_exac[0,:]-u_noise[0,:],'b-',alpha=0.5,label='noisy')
pl.grid()
pl.legend(loc=0)
pl.ylabel(r'$T(t)$')
pl.xlabel('Time (Seconds)')
# Temperaturas (del MAP, de la media, mediciones y exacta) en la frontera
pl.subplot(224)
pl.plot(fmap.t,u_exac[1,:]-u_map[1,:],'k-',label='map')
pl.plot(fmap.t,u_exac[1,:]-u_mean[1,:],'r-',label='mean')
#pl.plot(fmap.t,u_exac[1,:],'b-',label='exact')
pl.plot(fmap.t,u_exac[1,:]-u_noise[1,:],'b-',alpha=0.5,label='noisy')
pl.grid()
pl.legend(loc=0)
pl.xlabel('Time (Seconds)')
fig = pl.gcf()
plt.tight_layout()
fig.savefig("true_vs_estimators.png")

truth = fmap.diff_coeff_exact(fmap.tpars)[1:]
range = [(0.9*x,1.1*x) for x in truth]
figure = corner.corner(chain[:,:-1],truths=truth,range=range)
#figure = corner.corner(chain[:,:-1],truths=truth)
fig = pl.gcf()
fig.savefig("posterior_vs_truth.png")

pl.figure()
pl.hist(chain[:,-1],bins=50)#,range=(0.0,5.0))
pl.savefig("hyper_parameter.png")
