# -*- coding: utf-8 -*-
"""
Created on Tue May 31 18:21:31 2022

@author: antoi
"""

from math import pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

a = 54 # Satellites/an
b = 10 # Satellites/an
c = 1.935 # Radians/an
d = 0.1680+pi # Radians


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

t1 = np.linspace(2008,2022,2000)

X = [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]
Y = [36,43,37,43,40,46,50,44,42,51,67,66,82,111]

plt.figure(4)
plt.grid()
plt.title('Lancees de la terre')
plt.plot(X,Y, 'b')
plt.show()



def A(t):
    return a+b*np.sin(c*t+d)

def TN(t):
    return (f+g*np.sin(h*t+k))

def Tn(t):
    return (p+q*np.sin(h*t+k))


plt.figure(1)
plt.grid()
plt.title('Lancees de la terre')
plt.plot(t1, A(t1), 'g')
plt.show()

plt.figure(2)
plt.grid()
plt.title('Durée de vie d un fragment')
plt.plot(t1, Tn(t1), 'r')
plt.show()

plt.figure(3)
plt.grid()
plt.title('Durée de vie d un satellite')
plt.plot(t1, TN(t1), 'b')
plt.show()