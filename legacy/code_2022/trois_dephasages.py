# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 17:43:22 2022

@author: antoi
"""

from math import pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

a = 31.41 # Satellites/an
b = 7.794 # Satellites/an
c = 1.935 # Radians/an
d = 0.1680 # Radians


d2 = d + pi
d3 = pi/2
f = 14420 # Annéess
g = -10430 # Années
h = 0.5712 # Radians/an
k = -0.9996 # Radians
p = 184.9 # Années
q = -137.9 # Années 
x = 6.895e-10 # Années^-1.Fragment^-1
y = 1.369e-9 # Années^-1.Satellite^-1
z = 2.869e-14 # Années^-1.Fragment^-1
alpha = 10000  # Fragments/satellite
beta = 70 # Fragments/satellite
gamma = 56000 # Fragments/satellite
n0 = 110400 # Fragments
N0 = 4960 # Satellite

t1 = np.linspace(20,2044,2500)


def A1(t):
    return a+b*np.sin(c*t+d)

def A2(t):
    return a+b*np.sin(c*t+d2)

def A3(t):
    return a+b*np.sin(c*t+d3)

def calcul_TN(t):
    return f+g*np.sin(h*t+k)

def calcul_Tn(t):
    return p+q*np.sin(h*t+k)

plt.figure(1)
plt.grid()
plt.title('Lancées de la terre')
plt.plot(t1, A1(t1), 'g', t1, A3(t1), 'b', t1, A2(t1), 'r')
plt.show()


def systeme1(s: list,t,x,beta,alpha):
    A2, TN2, Tn2 = calcul_A1(t), calcul_TN(t), calcul_Tn(t)
    N, n = s[0], s[1]
    dN_dt = A2 -x*n*N -N/TN2  -2*y*N**2
    dn_dt = beta*A2 + alpha*x*n*N -n/Tn2 + gamma*y*N**2 -2*z**2
    
    return [dN_dt, dn_dt]

def systeme2(s: list,t,x,beta,alpha):
    A2, TN2, Tn2 = calcul_A2(t), calcul_TN(t), calcul_Tn(t)
    N, n = s[0], s[1]
    dN_dt = A2 -x*n*N -N/TN2  -2*y*N**2
    dn_dt = beta*A2 + alpha*x*n*N -n/Tn2 + gamma*y*N**2 -2*z**2
    
    return [dN_dt, dn_dt]

def systeme3(s: list,t,x,beta,alpha):
    A2, TN2, Tn2 = calcul_A3(t), calcul_TN(t), calcul_Tn(t)
    N, n = s[0], s[1]
    dN_dt = A2 -x*n*N -N/TN2  -2*y*N**2
    dn_dt = beta*A2 + alpha*x*n*N -n/Tn2 + gamma*y*N**2 -2*z**2
    
    return [dN_dt, dn_dt]

t1 = np.linspace(2022,2500,2500)

solution1 = odeint(systeme1,[N0,n0], t1, args=(x,beta,alpha))
solution2 = odeint(systeme2,[N0,n0], t1, args=(x,beta,alpha))
solution3 = odeint(systeme3,[N0,n0], t1, args=(x,beta,alpha))

N1 = solution1[:,0]
n1 = solution1[:,1]
N2 = solution2[:,0]
N3 = solution3[:,0]
n2 = solution2[:,1]
n3 = solution3[:,1]

plt.figure(1)
plt.title('Nombres de satellites intacts en orbite basse en fonction du temps')
plt.ylabel('Nombre de satellites')
plt.xlabel('Temps en années')
plt.xlim(2100,2102)
plt.ylim(6900, 7100)
plt.plot(t1,N1,'r',t1,N2, 'b', t1, N3, 'g')
plt.grid()
plt.show()

"""
plt.figure(2)
plt.title('Fragments en orbite basse en fonction du temps')
plt.ylabel('Nombre de fragments')
plt.xlabel('Temps en années')
plt.xlim(2220,2230)
plt.ylim(6.5e7,6.6e7)
plt.plot(t1,n1,'r', t1,n2, 'b', t1,n3, 'g')
plt.grid()
plt.show()

"""
