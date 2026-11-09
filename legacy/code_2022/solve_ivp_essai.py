# -*- coding: utf-8 -*-
"""
Created on Thu May 19 21:14:01 2022

@author: antoi
"""
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
from math import exp,cos,sin
from scipy.integrate import solve_ivp
from scipy.integrate import odeint


def euler(h,T,t0,v0,f):
    list_t = [t0]
    list_v = [v0]
    v = v0
    t = t0
    while t < T:
        v+=h*f(t,v)
        t+=h
        list_t.append(t)
        list_v.append(v)
    return list_t, list_v

a = 31.41 # Satellites/an
b = 7.794 # Satellites/an
c = 1.935 # Radians/an
d = 0.1680 # Radians
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
N0 = 4650 # Satellite
A = 30.5 #Satellites/an
t1 = np.linspace(2000,2500,250)

def systeme(t,S):
    N = S[0]
    n = S[1]
    dN_dt = A -x*n*N
    dn_dt = beta*A -alpha*x*n*N
    
    return [dN_dt, dn_dt]

solution = solve_ivp(systeme, [2000,2500], [N0,n0], method= 'RK45')

N = solution[:,0]
n = solution[:,1]


plt.figure(1)
plt.title('Nombres de satellites intacts en orbite basse en fonction du temps')
ax = plt.axes
ax = ax.set(xlabel = 'Temps en années', ylabel ='Nombre de satellites intacts')
plt.plot(N,t1)
plt.show()