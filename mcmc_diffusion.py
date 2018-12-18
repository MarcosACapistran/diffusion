#coding: utf8

from scipy import interpolate as ip
from scipy import special
import numpy as np
import scipy as sp
import pickle
import time
import pytwalk
import os
from forward_mapping import *

"""
Este programa implementa el algoritmo MCMC para identificar el parámetro de
difusividad térmica k. Los resultados se guardan en el directorio output0.
Se trabaja con dos propuestas de k. Una cóncava y una sigmoide. Para elegir
entre ellas hay que descomentar/comentar las líneas adecuadas: descomentar 25/29
para la cóncava (al revés para la sigmoide). Además, se puede elgir el tipo de a 
priori: 
"""
    
# set the state of random
#np.random.seed(seed=100)

# Inicializa el mapeo de parámetros a datos
"""
k cóncava
"""
fmap = forward_mapping()
"""
k sigmoide
"""
#fmap = forward_mapping(pars0 = 1.1725473550631119e-06)

# Datos sintéticos en un mallado fino
#data,noisy_data = fmap.create_data_coarse()
data,noisy_data = fmap.create_data()

# matriz del grafo con pesos de la norma H1
A_graph = np.zeros((fmap.npars+1,fmap.npars+1))
i,j = np.indices(A_graph.shape)
A_graph[i==j] = 2.0
A_graph[i==j-1] = -1.0
A_graph[i==j+1] = -1.0
A_graph /= (fmap.taupars)**2
# matriz de covarianza
Sigma = np.linalg.inv(A_graph)
# matriz de covarianza condicional a k0 fijo
mu_bar = Sigma[1:,0]*(1.0/Sigma[0,0])*(fmap.pars0) # mu=0 is omitted
Sigma_bar = Sigma[1:,1:]-(1.0/Sigma[0,0])*np.outer(Sigma[1:,0],Sigma[1:,0])
Sigma_bar_inv = np.linalg.inv(Sigma_bar)

# constante de normalización de la verosimilitud
cte_like = -((fmap.nt+1.0)/2.0)*np.log(2.0*np.pi*fmap.std**2)

# parametrización y constantes de normalización de la gamma
k1 = 1.0
theta1 = 1.0
cte_gamma = -np.log(special.gamma(k1)*theta1**k1)

# la constante de normalización del campo aleatorio Gaussiano 
# se evalúa dentro de la energía

# Logaritmo de la distribución posterior
def energy(theta_aux):    
    theta = np.insert(theta_aux,0,np.log(fmap.pars0))
    my_fsc = fmap.fscub(np.exp(theta[:-1]),0)
    my_k = my_fsc(fmap.t)       
    my_soln = fmap.solve(my_k)
    """
    Verosimilitud Gaussiana
    """
    log_likelihood = cte_like-0.5*(np.linalg.norm(noisy_data[1:]-my_soln[1:]))**2/fmap.std**2
    """
    log de la densidad del hiper-parámetro
    """
    a1 = cte_gamma+(k1-1)*np.log(theta[-1])-(theta[-1]/theta1) 
    """
    A priori: campo aleatorio Gaussiano
    """
    A_prec = Sigma_bar_inv/(2.0*theta[-1]**2) # matriz de precisión
    A_cov = (2.0*theta[-1]**2)*Sigma_bar # matriz de covarianza
    cte_gmrf = -0.5*np.log(np.linalg.det(2.0*np.pi*A_cov))    
    log_prior = cte_gmrf-0.5*np.dot(theta[1:-1]-mu_bar,np.dot(A_prec,theta[1:-1]-mu_bar))
    return -log_likelihood-log_prior-a1


# Soporte de Q
def support_Q(theta_aux):
    rt = True
    theta = np.insert(theta_aux,0,np.log(fmap.pars0))
    my_fsc_log = fmap.fscub(theta[:-1],0)
    my_fsc = fmap.fscub(np.exp(theta[:-1]),1)    
    my_u = my_fsc_log(fmap.t)
    my_k = my_fsc(fmap.t)    
    num = np.diff(my_k)
    den = np.diff(my_u)
    rt &= fmap.tau*(np.sum(num/den))<fmap.C
    rt &= np.all(np.diff(my_u)/(2.0*fmap.tau)<fmap.fimed)
    rt &= np.all(np.diff(my_k)>=0.0)
    rt &= theta[-1]<1.0*10.0**0
    rt &= theta[-1]>0.0         
    return rt

# Puntos iniciales
def init0():
    """
    descomentar lineas 113 a 118 para iniciar cerca del MAP
    descomentar lineas 119 a 120 para iniciar cerca de K0
    """
    # pars = fmap.diff_coeff_exact(fmap.tpars)[1:]
    # std = np.mean(pars)*10.0**-5
    # noise = np.random.randn(fmap.npars)
    # pars += std*noise
    # pars = np.log(pars)
    # pars = np.insert(pars,fmap.npars,1.0*10.0**2*np.random.uniform())
    pars = np.sort(np.random.uniform(low=np.log(fmap.pars0),high=np.log(2.0*fmap.pars0),size=fmap.npars))
    pars = np.insert(pars,fmap.npars,1.0*10.0**0*np.random.uniform())      
    return pars

# Marco del problema inverso
Tmax = 100000 # Número de pasos del MCMC
ndim = fmap.npars+1 # Dimensión del espacio de parámetros

# Revisa que existe el directorio donde se escribirán los resultados
directory = "output0"
if not os.path.exists(directory):
    os.makedirs(directory)

# Abre un archivo y guarda el marco del problema inverso    
setting = {'Tmax':Tmax,'ndim':ndim,'std':fmap.std,'u':data,'v':noisy_data}
pickle.dump(setting,open("inv_prob_setting.pk","wb"))

# Abre un archivo para guardar los resultados
f = open(directory+"/chain.dat","w")
f.close()
f = open(directory+"/probs.dat","w")
f.close()

# inicializa las cadenas de muestreo
k0 = init0()
kp0 = init0()
e_k0 = energy(k0)
e_kp0 = energy(kp0)    

# guarda las condiciones iniciales
f = open(directory+"/chain.dat","a")
value = ' '.join([str(x) for x in k0])
f.write(' '+value+' ')
f.close()

# guarda las condiciones iniciales
f = open(directory+"/probs.dat","a")
value = ''.join(str(e_k0))
f.write(' '+value+' ')
f.close()

# inicializa la clase twalk
my_walk = pytwalk.pytwalk(n=ndim,U=energy,Supp=support_Q)

t0 = time.time() 
# Ejecute el single variable exchange algorithm
for i in np.arange(1,Tmax):
    # update = [y, yp, ke, A, u_prop, up_prop]
    # y, yp: the proposed jump
    # ke: The kernel used, 0=nothing, 1=Walk, 2=Traverse, 3=Blow, 4=Hop
    # A: the M-H ratio
    # u_prop, up_prop: The values for the objective func. at the proposed jumps
    update = my_walk.onemove(k0,e_k0,kp0,e_kp0)
    # hace un paso del single exchange variable algorithm
    if support_Q(update[0])&support_Q(update[1]):
        A_prec_old = Sigma_bar_inv/(2.0*k0[-1]**2) # matriz de precisión vieja
        A_prec_new = Sigma_bar_inv/(2.0*update[0][-1]**2) # matriz de precisión nueva
        A_cov_old = (2.0*k0[-1]**2)*Sigma_bar # matriz de covarianza vieja
        A_cov_new = (2.0*update[0][-1]**2)*Sigma_bar # matriz de covarianza nueva
        x = np.random.multivariate_normal(mu_bar,A_cov_new) # variable intermedia
        cte_gmrf_old = -((fmap.npars)/2.0)*np.log(2.0*np.pi)-0.5*np.log(np.linalg.det(A_cov_old))
        cte_gmrf_new = -((fmap.npars)/2.0)*np.log(2.0*np.pi)-0.5*np.log(np.linalg.det(A_cov_new))
        #log_prior_old = cte_gmrf_old-0.5*np.dot(k0[:-1]-mu_bar,np.dot(A_prec_old,k0[:-1]-mu_bar))
        log_prior_old = cte_gmrf_old-0.5*np.dot(x-mu_bar,np.dot(A_prec_old,x-mu_bar))
        log_prior_new = cte_gmrf_new-0.5*np.dot(x-mu_bar,np.dot(A_prec_new,x-mu_bar))
        a = update[3]*np.exp(log_prior_old-log_prior_new)
        if np.random.uniform() < a:
            k0 = np.copy(update[0])
            e_k0 = np.copy(update[4])
            kp0 = np.copy(update[1])
            e_kp0 = np.copy(update[5])
        f = open(directory+"/chain.dat","a")
        value = ' '.join([str(x) for x in k0])
        f.write(' '+value+' ')
        f.close()
        f = open(directory+"/probs.dat","a")
        value = ''.join(str(e_k0))
        f.write(' '+value+' ')
        f.close()
t1 = time.time()-t0
print ('time spent:',t1)
