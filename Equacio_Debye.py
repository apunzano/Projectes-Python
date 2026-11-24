# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 14:31:27 2025

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.stats import linregress
from tabulate import tabulate

# Given constants
a1 = 87.9
b1 = 0.404
c1 = 9.59e-4
d1 = 1.33e-6
a2 = 80.7
b2 = 4.42e-3
c2 = 1.37e-13
d2 = 651
To = 133

# We also define the given functions
def epsilon_o(T):
    """Dielectric permitivity estàtica en funció de T"""
    return a1 - b1*T + c1*T**2 - d1*T**3

def epsilon_inf(T):
    """Permitivitat a freqüències infinites en funció de T"""
    return epsilon_o(T) - a2*np.exp(-b2*T)

def tau(T):
    """ Temps de relaxacio rotacional"""
    return c2*np.exp(d2/(T+To))

def epsilon(freq):
    """Permitivitat dielectrica en funció de la freqüència"""
    return epsilon_inf(T) + (epsilon_o(T)-epsilon_inf(T))/(1 - 1j*2*np.pi*freq*tau(T))

def vt(freq):
    """Diferència entre la part imaginària i real de la permitivitat"""
    return np.imag(epsilon(freq))- np.real(epsilon(freq))



T = 0 # Temperatura inicial en ºC
a = 0
freq = np.logspace(8, 12) # Freqüència
freqt = np.empty([5,2]) # Freqüència de tall
y = np.empty([50,5], dtype=np.complex128) # Array per emmagatzemar les permitivitats
resultats_freq = []
resultats_max = []

# Bucle principal que varia la temperatura de 20 en 20 graus fins a 100°C
while T < 100:
    T = T + 20
    n = 0
    
    # Bucle per calcular la permitivitat per a cada freqüència
    while n < freq.size:
        sol = epsilon(freq[n])
        y[n, a] = sol
        n = n + 1
        
    # APARTAT 2 i 3: Trobar la freqüència on la part imaginària és màxima
    sol= fsolve(vt,1e10)
    freqt[a,0] = sol[0]
    freqt[a,1] = T
    maxim = np.argmax(np.imag(y[:, a]))
    resultats_max.append([
        f"{T}°C",
        f"{maxim:.3e}Hz",
        f"{sol[0]:.3e}Hz"])
    
    # APARTAT 3: Trobar la freqüència on la part imaginària i la real tenen
    a = a + 1


# Imprimim en consola els resultats de l'apartat 2
print("TAULA 1: MÀXIMS DE ε IMAGINARIA I FREQ. DE TALL")
headers_frecuencias = ["Temperatura", "Freqüència on εi max", "Freqüència de Tall"]
print(tabulate(resultats_max, headers=headers_frecuencias, 
               tablefmt="fancy_grid", stralign="center"))

# Separem la part imaginària de la real de ε
imagy = np.imag(y)
realy = np.real(y)

# PRIMERA FIGURA: permitivitat en relació a la freqüència
plt.figure()
plt.semilogx()

plt.grid(True)
plt.xlabel("Freqüència (Hz)")
plt.ylabel(r'$\epsilon$')


colors = ['r', 'y', 'g', 'c', 'm']
labels = [f'{T}°C' for T in freqt[:,1]]


for i in range(5):
    # Parts reals per a diferents temperatures
    plt.plot(freq, realy[:, i], colors[i], label= f'({labels[i]})')
    # Parts imaginàries per a diferents temperatures
    plt.plot(freq, imagy[:, i], colors[i])
    
plt.legend()


# Regressió lineal per trobar la relació entre freqüència de tall i temperatura
reg = linregress(freqt[:,0], freqt[:,1], alternative='greater')
print("\n")
print(f"En la recta de regressió y = a + bx, a={reg[1]:.4g} b={reg[0]:.4g} i el coeficient de corelació és r = {1}.")


# SEGONA FIGURA: temperatura en relació a la freqüència de tall
plt.figure()
plt.xlim(1e10, 0.9e11)
plt.grid(True)
plt.xlabel("Freqüència de tall (Hz)")
plt.ylabel("T (ºC)")
plt.plot(freqt[:,0], freqt[:,1],marker='o', ls=" ")
plt.plot(freqt[:,0], reg[1]+reg[0]*freqt[:,0], label=f"y = {reg[1]:.4g} + {reg[0]:.4g} x")
plt.legend()





