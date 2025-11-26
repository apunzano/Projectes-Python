# -*- coding: utf-8 -*-

import numpy as np
import scipy.integrate
import scipy.special
from math import factorial
import matplotlib.pyplot as plt

# Constants donades
A = 1.35
m = 1
g = 1
L = 1
b = 0.5
Omega = 0.666

def pendol(y, t):
    """Equació del pèndol forçat amb fricció"""
    theta, omega = y
    dtheta = omega
    domega = -(g/L) * np.sin(theta) - (b/m) * omega + (A/(m*L)) * np.cos(Omega*t)
    return [dtheta, domega]


t2 = np.linspace(0, 100, 10000)
ci2 = np.array([0, 0.1])  # [angle inicial, velocitat angular inicial]
sol2 = scipy.integrate.odeint(pendol, ci2, t2)

# Representació de la velocitat angular i l'angle respecte al temps
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5))

# Angle
ax1.plot(t2, sol2[:, 0])
ax1.set_xlabel("t (s)")
ax1.set_ylabel(r'$\theta$ (rad)')

# Velocitat angular
ax2.plot(t2, sol2[:, 1])
ax2.set_xlabel("t (s)")
ax2.set_ylabel(r'$\omega$ (m/s)')

fig.suptitle('PÈNDOL ESMORTEÏT I FORÇAT', fontsize=16)
plt.tight_layout()
plt.show()

# La representació gràfica dels angles pot resultar confusa, 
# ja que estem acostumats a visualitzar-los en l'interval [0, 2π] radians.