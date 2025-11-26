# -*- coding: utf-8 -*-

import numpy as np
import scipy.integrate
import scipy.special
from math import factorial
import matplotlib.pyplot as plt

# Constants donades
B = 1
valors_n = [0, 2, 4]

def schrodinger(y, x):
    """Sistema d'equacions de Schrödinger"""
    psi, dpsi = y
    d2psi = -psi * B * ((2 * n + 1) - B * x**2)
    return [dpsi, d2psi]

def psi_n(n, x):
    """Funció d'ona analítica per a l'oscil·lador harmònic quàntic"""
    H_n = scipy.special.eval_hermite(n, x)  # Polinomi d'Hermite
    psi = (np.exp(-x**2 / 2) * H_n) / np.sqrt(2**n * factorial(n) * np.sqrt(np.pi))
    return psi

x = np.linspace(0, 5)
xtot = np.linspace(-5, 5, 99)
ci = ([0.751126, 0], [-0.531125, 0], [0.459969, 0])  # Condicions inicials per cada n

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

# Solució analítica
for n in valors_n:
    ax1.plot(xtot, psi_n(n, xtot), label=f'n = {n}')
ax1.set_title("Solució Analítica")
ax1.set_xlabel("x")
ax1.set_ylabel(r'$\psi$ ($x$)')
ax1.legend()

# Solució numèrica
for n in valors_n:
    sol2_1 = scipy.integrate.odeint(schrodinger, ci[valors_n.index(n)], x)
    sol2_2 = scipy.integrate.odeint(schrodinger, ci[valors_n.index(n)], -x)
    sol2 = np.concatenate([sol2_2[::-1], sol2_1[1:]])
    ax2.plot(xtot, sol2[:, 0], label=f'n = {n}')
ax2.set_title("Amb Equació Diferencial")
ax2.set_xlabel("x")
ax2.set_ylabel(r'$\psi$ ($x$)')
ax2.legend()

fig.suptitle('EQ. SCHRÖDINGER')
plt.tight_layout()
plt.show()