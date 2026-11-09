# -*- coding: utf-8 -*-
"""
Created on Thu May 26 16:41:58 2022

@author: antoi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

a = 31.41 # Satellites/an
b = 7.794 # Satellites/an
c = 1.935 # Radians/an
d = 0.1680 # Radians

f = 14420 # Annéess
g = -10430 # Années
p = 184.9 # Années
q = -137.9 # Années
h = 0.5712 # Radians/an
k = -0.9996 # Radians
 

x = 6.895e-10 # Années^-1.Fragment^-1
y = 1.369e-9 # Années^-1.Satellite^-1
z = 2.869e-14 # Années^-1.Fragment^-1
alpha = 10000  # Fragments/satellite
beta = 70 # Fragments/satellite
gamma = 56000 # Fragments/satellite
n0 = 0 # Fragments
N0 = 4960 # Satellite




def calcul_A(t):
    return a+b*np.sin(c*t+d)

def calcul_TN(t):
    return f+g*np.sin(h*t+k)

def calcul_Tn(t):
    return p+q*np.sin(h*t+k)


def systeme(s: list,t,x,beta,alpha):
    A2, TN2, Tn2 = calcul_A(t), calcul_TN(t), calcul_Tn(t)
    N, n = s[0], s[1]
    dN_dt = A2 -x*n*N -2*y*N**2 - N/TN2
    dn_dt = beta*A2 + alpha*x*n*N -n/Tn2 + gamma*y*N**2 -2*z**2
    
    return [dN_dt, dn_dt]

t1 = np.linspace(2022, 2500,2500)

solution = odeint(systeme,[N0,n0], t1, args=(x,beta,alpha))

N1 = solution[:,0]
n1 = solution[:,1]

plt.figure(1)
plt.title('Nombres de satellites intacts sans rentree atmospherique')
plt.ylabel('Nombre de satellites')
plt.xlabel('Temps en années')
plt.plot(t1,N1,'r')
plt.grid()
plt.show()

plt.figure(2)
plt.title('Fragments en orbite basse en fonction du temps')
plt.ylabel('Nombre de fragments')
plt.xlabel('Temps en années')
plt.plot(t1,n1,'g')
plt.grid()
plt.show()

