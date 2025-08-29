"""
1D Harmonic Oscillator - Time Dependent Schrodinger Equation (TDSE)

This script solves the TDSE by using an algorithm called crank nicolson method.
The Hamiltonian is H = T + V, with T built from second-order central-difference 
Laplacians and V a real (static or time-dependent) potential sampled on the grid.
A single guassian wave packet is used to show how it would move under a quantum 
harmonic oscillator and the script then plots the entire wavefunction with its real
and its imaginary component. 
"""

import numpy as np
import sys, os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

current_dir = os.path.dirname(os.path.abspath(__file__))    
project_root = os.path.abspath(os.path.join(current_dir, ".."))  
sys.path.insert(0, project_root)

from Master.TDSE import crank_nicolson_tdse_1D

# ---------------- Grid and Potential Setup -------------
N   = 1200
xmax = 10.0
x   = np.linspace(-xmax, xmax, N)
dx  = x[1] - x[0]

ħ = 1.0
m = 1.0
ω = 1.0
Vx = 0.5 * (ω**2) * x**2

# --------------- Moving Guassian Packet Setup --------------
x0    = -2.0        
sigma = 1.0       
p0    = 2.0         
psi0  = (1.0/(np.pi*sigma**2)**0.25) * np.exp(-(x-x0)**2/(2*sigma**2)) * np.exp(1j*p0*x)
psi0 /= np.sqrt(np.trapezoid(np.abs(psi0)**2, x))

# ----------------- TDSE Solver ------------------
dt      = 0.002        
nsteps  = 5000          
save_k  = 10             

Psi_t, times = crank_nicolson_tdse_1D(x, Vx, psi0, dt, nsteps, ħ=ħ, m=m, store_every=save_k)

# -------------- animate Re, Im, and |ψ| ---------------
fig, ax = plt.subplots()
line_re, = ax.plot([], [], label='Re[ψ]', lw=1.5, color='cyan')
line_im, = ax.plot([], [], label='Im[ψ]', lw=1.5, color='magenta')
line_abs,= ax.plot([], [], label='|ψ|',  lw=2.0, color='yellow')

ax.set_xlim(x.min(), x.max())
ax.set_ylim(-1.2*np.max(np.abs(Psi_t)), 1.2*np.max(np.abs(Psi_t)))
ax.set_xlabel('x')
ax.set_ylabel('Amplitude')
title = ax.set_title('')
ax.legend(loc='upper right')

ax.legend(loc='upper right')
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.tick_params(colors='white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.title.set_color('white')
for spine in ax.spines.values():
    spine.set_edgecolor('white')
for text in ax.legend().get_texts():
    text.set_color('white')


def init():
    line_re.set_data([], [])
    line_im.set_data([], [])
    line_abs.set_data([], [])
    return line_re, line_im, line_abs, title

def update(i):
    psi = Psi_t[i]
    line_re.set_data(x, np.real(psi))
    line_im.set_data(x, np.imag(psi))
    line_abs.set_data(x, np.abs(psi))
    title.set_text(f"1D Harmonic Oscillator for t = {times[i]:.3f}")
    return line_re, line_im, line_abs, title

ani = FuncAnimation(fig, update, frames=len(times), init_func=init, blit=False, interval=30)
ani.save("1DHarmonicOscillator.gif", writer="pillow", fps=44)
plt.show()
