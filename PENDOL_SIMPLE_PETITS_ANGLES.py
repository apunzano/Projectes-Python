# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 12:55:04 2025

"""

import numpy as np
import scipy.integrate
import scipy.special
import matplotlib.pyplot as plt

g = 9.81
L = 1

def eq_dif(y, t):
    """
    Sistema d'equacions diferencials per al pèndol simple
    y[0] = velocitat angular
    y[1] = angle
    """
    return -g * y[1] / L, y[0]

t1 = np.linspace(0, 10, 100)
ci1 = np.array([0, 10])  # Condicions inicials
sol1 = scipy.integrate.odeint(eq_dif, ci1, t1) # Resolució de l'equació diferencial

# Gràfiques angle i velocitat angular
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5))

# Velocitat angular
ax1.plot(t1, sol1[:, 0])
ax1.set_xlabel("t (s)")
ax1.set_ylabel(r'$\omega$ (m/s)')

# Angle
ax2.plot(t1, sol1[:, 1])
ax2.set_xlabel("t (s)")
ax2.set_ylabel(r'$\theta$ (rad)')

fig.suptitle('PÈNDOL SIMPLE', fontsize=16)
plt.tight_layout()
plt.show()
