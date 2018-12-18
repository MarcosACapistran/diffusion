import pylab as pl
import sys
import numpy as np
import matplotlib
#matplotlib.style.use('bmh')
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

'''
grafica menos el logaritmo de las muestras de la distribucion posterior del MCMC
'''
x1 = np.int(sys.argv[1]) # indice de la iteracion desde donde quieres graficar
f = open("output0/probs.dat", "r")
a = f.read()
b = a.replace('[','')
c = b.replace(']','')
d = c.replace('inf','0.0 ')
a = d.split()
a = list(map(float,a))
a = np.asarray(a)
a = -a[x1:]
pl.figure(figsize=(8,6))
pl.plot(a,'k',alpha=0.5) # una grafica del progreso de la cadena de Markov
pl.locator_params(nbins=10, axis='x')
pl.grid()
plt.tight_layout()
pl.savefig("trace_plot.png")

