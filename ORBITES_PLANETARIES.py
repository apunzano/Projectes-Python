# -*- coding: utf-8 -*-

import numpy as np
import scipy.integrate
import scipy.special
import matplotlib.pyplot as plt

# Constants donades
G = 6.67e-11  # Constant gravitacional (SI)
M = 1.9891e30  # Massa del Sol (Kg)
au = 1.49598e11  # Unitat astronòmica (m)

# TERRA
aph_terra = 1.0167  # au (afeli)
velap = 29.29e3  # m/s (velocitat a l'afeli)
T = 3.15576e7  # Període orbital (1 any en segons)

# HALLEY
aph_halley = 35.1  # au
vel_aphelion = 0.91e3  # m/s
T_halley = 3.15576e7 * 75.3  # Període orbital (75.3 anys en segons)

ci4 = [aph_terra * au, 0, 0, velap]  # Condicions inicials Terra
ci5 = [aph_halley * au, 0, 0, vel_aphelion]  # Condicions inicials Halley
t4 = np.linspace(0, T, 1000)
t5 = np.linspace(0, T_halley, 1000)

def sist_acoblat(z, t):
    """Sistema d'equacions per a òrbites planetàries"""
    x, vel_x, y, vel_y = z
    d2x = -G * M * x / (x**2 + y**2)**(3/2)
    d2y = -G * M * y / (x**2 + y**2)**(3/2)
    return [vel_x, d2x, vel_y, d2y]

sol4 = scipy.integrate.odeint(sist_acoblat, ci4, t4)  # Terra
sol5 = scipy.integrate.odeint(sist_acoblat, ci5, t5)  # Halley

# Gràfiques per a la Terra
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
ax1.plot(t4, sol4[:, 0])
ax1.set_xlabel("t")
ax1.set_ylabel("x(t)")
ax2.plot(t4, sol4[:, 2])
ax2.set_xlabel("t")
ax2.set_ylabel("y(t)")
ax3.plot(t4, np.sqrt(sol4[:, 1]**2 + sol4[:, 3]**2), label='$v(t)$')
ax3.set_xlabel("t")
ax3.set_ylabel("r(t)")
ax4.plot(sol4[:, 0], sol4[:, 2])
ax4.set_xlabel("x")
ax4.set_ylabel("y(x)")
ax4.axis('equal')


fig.suptitle('PLANETA TERRA', fontsize=16)
plt.tight_layout()
plt.show()

# Gràfiques per al Halley
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
ax1.plot(t5, sol5[:, 0])
ax1.set_xlabel("t")
ax1.set_ylabel("x(t)")
ax2.plot(t5, sol5[:, 2])
ax2.set_xlabel("t")
ax2.set_ylabel("y(t)")
ax3.plot(t5, np.sqrt(sol5[:, 1]**2 + sol5[:, 3]**2), label='$v(t)$')
ax3.set_xlabel("t")
ax3.set_ylabel("r(t)")
ax4.plot(sol5[:, 0], sol5[:, 2], label='$y(x)$')
ax4.set_xlabel("x")
ax4.set_ylabel("y(x)")
ax4.axis('equal')


fig.suptitle('COMETA HALLEY', fontsize=16)
plt.tight_layout()
plt.show()