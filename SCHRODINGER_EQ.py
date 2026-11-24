# This code compares the analytical and numerical solutions of the Schrödinger equation 
# for a quantum harmonic oscillator. It calculates and plots the wave functions for 
# different quantum numbers to illustrate the agreement between both methods.

# -*- coding: utf-8 -*-

import numpy as np
import scipy.integrate
import scipy.special
from math import factorial
import matplotlib.pyplot as plt

# Given data
B = 1
n_values = [0, 2, 4]

def schrodinger(y, x):
    """Schrödinger equation system"""
    psi, dpsi = y
    d2psi = -psi * B * ((2 * n + 1) - B * x**2)
    return [dpsi, d2psi]

def psi_n(n, x):
    """Analytical wave function for the quantum harmonic oscillator"""
    H_n = scipy.special.eval_hermite(n, x)  # Hermite polynomial
    psi = (np.exp(-x**2 / 2) * H_n) / np.sqrt(2**n * factorial(n) * np.sqrt(np.pi))
    return psi

x = np.linspace(0, 5)
x_total = np.linspace(-5, 5, 99)
initial_conditions = ([0.751126, 0], [-0.531125, 0], [0.459969, 0])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

# Analytical solution
for n in n_values:
    ax1.plot(x_total, psi_n(n, x_total), label=f'n = {n}')
ax1.set_title("Analytical Solution")
ax1.set_xlabel("x")
ax1.set_ylabel(r'$\psi$ ($x$)')
ax1.legend()

# Numerical solution
for n in n_values:
    sol2_1 = scipy.integrate.odeint(schrodinger, initial_conditions[n_values.index(n)], x)
    sol2_2 = scipy.integrate.odeint(schrodinger, initial_conditions[n_values.index(n)], -x)
    sol2 = np.concatenate([sol2_2[::-1], sol2_1[1:]])
    ax2.plot(x_total, sol2[:, 0], label=f'n = {n}')
ax2.set_title("Using the Differential Equation")
ax2.set_xlabel("x")
ax2.set_ylabel(r'$\psi$ ($x$)')
ax2.legend()

fig.suptitle('SCHRÖDINGER EQUATION')
plt.tight_layout()
plt.show()
