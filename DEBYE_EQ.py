# This script computes the complex dielectric permittivity of a material 
# over a frequency range using a Debye relaxation model. Identifies the 
# frequencies of maximum imaginary permittivity and the cutoff frequency 
# and performs a linear regression between cutoff frequency and temperature.

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
    """Static dielectric permittivity as a function of T"""
    return a1 - b1*T + c1*T**2 - d1*T**3

def epsilon_inf(T):
    """Permittivity at infinite frequencies as a function of T"""
    return epsilon_o(T) - a2*np.exp(-b2*T)

def tau(T):
    """ Rotational relaxation time"""
    return c2*np.exp(d2/(T+To))

def epsilon(freq):
    """Dielectric permittivity as a function of frequency"""
    return epsilon_inf(T) + (epsilon_o(T)-epsilon_inf(T))/(1 - 1j*2*np.pi*freq*tau(T))

def vt(freq):
    """Difference between the imaginary and real parts of the permittivity"""
    return np.imag(epsilon(freq))- np.real(epsilon(freq))



T = 0 # Initial temperature in ºC
a = 0
freq = np.logspace(8, 12) # Frequency
freqt = np.empty([5,2]) # Cutoff frequency
y = np.empty([50,5], dtype=np.complex128) # Array to store the permittivities
resultats_freq = []
resultats_max = []

# Main loop that varies the temperature from 20 to 100°C in steps of 20 degrees
while T < 100:
    T = T + 20
    n = 0
    
    # Loop to calculate the permittivity for each frequency
    while n < freq.size:
        sol = epsilon(freq[n])
        y[n, a] = sol
        n = n + 1
        
    # PART 2 and 3: Find the frequency where the imaginary part is maximum
    sol= fsolve(vt,1e10)
    freqt[a,0] = sol[0]
    freqt[a,1] = T
    maxim = np.argmax(np.imag(y[:, a]))
    resultats_max.append([
        f"{T}°C",
        f"{maxim:.3e}Hz",
        f"{sol[0]:.3e}Hz"])
    
    # PART 3: Find the frequency where the imaginary part and the real part have
    a = a + 1


# Print the results of part 2 to the console
print("TABLE 1: MAXIMA OF ε IMAGINARY AND CUTOFF FREQ.")
headers_frecuencias = ["Temperature", "Frequency where εi max", "Cutoff Frequency"]
print(tabulate(resultats_max, headers=headers_frecuencias, 
               tablefmt="fancy_grid", stralign="center"))

# Separate the imaginary part from the real part of ε
imagy = np.imag(y)
realy = np.real(y)

# FIRST FIGURE: permittivity versus frequency
plt.figure()
plt.semilogx()

plt.grid(True)
plt.xlabel("Frequency (Hz)")
plt.ylabel(r'$\epsilon$')


colors = ['r', 'y', 'g', 'c', 'm']
labels = [f'{T}°C' for T in freqt[:,1]]


for i in range(5):
    # Real parts for different temperatures
    plt.plot(freq, realy[:, i], colors[i], label= f'({labels[i]})')
    # Imaginary parts for different temperatures
    plt.plot(freq, imagy[:, i], colors[i])
    
plt.legend()


# Linear regression to find the relationship between cutoff frequency and temperature
reg = linregress(freqt[:,0], freqt[:,1], alternative='greater')
print("\n")
print(f"In the regression line y = a + bx, a={reg[1]:.4g} b={reg[0]:.4g} and the correlation coefficient is r = {1}.")


# SECOND FIGURE: temperature versus cutoff frequency
plt.figure()
plt.xlim(1e10, 0.9e11)
plt.grid(True)
plt.xlabel("Cutoff frequency (Hz)")
plt.ylabel("T (ºC)")
plt.plot(freqt[:,0], freqt[:,1],marker='o', ls=" ")
plt.plot(freqt[:,0], reg[1]+reg[0]*freqt[:,0], label=f"y = {reg[1]:.4g} + {reg[0]:.4g} x")
plt.legend()
