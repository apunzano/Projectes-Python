# -*- coding: utf-8 -*-

import numpy as np
import scipy.integrate
import scipy.special
import matplotlib.pyplot as plt

# Constants donades
m = 1
k_1 = 10
k_2 = 0.5

ci3 = [1, 0, 0, 0]  # [x1, v1, x2, v2]
t3 = np.linspace(0, 40, 300)

def sist_acoblat(y, x):
    """Sistema de dos oscil·ladors acoblats"""
    x_1, vel_1, x_2, vel_2 = y
    d2x_1 = (-(k_1 + k_2) * x_1 + k_2 * x_2) / m
    d2x_2 = (-(k_1 + k_2) * x_2 + k_2 * x_1) / m
    return [vel_1, d2x_1, vel_2, d2x_2]

sol3 = scipy.integrate.odeint(sist_acoblat, ci3, t3)

# Gràfic de velocitats i posicions respecte al temps
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))

ax1.plot(t3, sol3[:, 0], label='$x_1(t)$')
ax1.plot(t3, sol3[:, 2], label='$x_2(t)$')
ax1.set_ylabel("x(t)")
ax1.legend()

ax2.plot(t3, sol3[:, 1], label='$v_1(t)$')
ax2.plot(t3, sol3[:, 3], label='$v_2(t)$')
ax2.set_xlabel("t")
ax2.set_ylabel("v(t)")
ax2.legend()

fig.suptitle('SISTEMA ACOBLAT', fontsize=16)
plt.tight_layout()
plt.show()
