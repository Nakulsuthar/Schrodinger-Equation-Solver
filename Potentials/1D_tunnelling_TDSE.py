"""
1D Tunnelling - Time Dependent Schrodingers Equation (TDSE)

This script solves the TDSE for a square barrier potential instead 
of a quantum harmonic oscillator. The setup is the same as 
1D_Harmonic_Oscillator_TDSE.py except a square barrier is introduced. 
This script shows an animation of how the wave packet moves towards the 
barrier and shows how some of the packet is reflected while the remanining 
is transmitted. This script also shows a spacetime heat map across time.
"""


import numpy as np
import sys, os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


current_dir = os.path.dirname(os.path.abspath(__file__))    
project_root = os.path.abspath(os.path.join(current_dir, ".."))  
sys.path.insert(0, project_root)

from Master.TDSE import crank_nicolson_tdse_1D

# -------------- Grid and Potential Setup ----------------

N = 4000
xmax = 100.0
x = np.linspace(-xmax,xmax,N)
dx = x[1] - x[0]

V0, a = 2.0,2.0
Vx = np.zeros_like(x)              
Vx[np.abs(x) < a/2] = V0


# ---------------- Moving Guassian Packet Setup ------------------
x0 = -20.0
sigma = 5.0
p0 = 2.0                          
psi0 = (1.0/(np.pi*sigma**2)**0.25) * np.exp(-(x-x0)**2/(2*sigma**2)) * np.exp(1j*p0*x)
psi0 /= np.sqrt(np.trapezoid(np.abs(psi0)**2,x))

# ----------------- TDSE Solver --------------------
dt = 0.005
nsteps = 22000
save_k = 10

Psi_t, times = crank_nicolson_tdse_1D(x, Vx, psi0, dt, nsteps,ħ=1.0, m=1.0, store_every=save_k)


# -------- Quantum Tunnelling Animation  -------- 
fig, ax = plt.subplots()
line_prob, = ax.plot([], [], lw=2, label=r'$|\psi(x,t)|^2$', color='Cyan')
ax.set_xlim(x.min(), x.max())
ax.set_ylim(0, 1.2*np.max(np.abs(Psi_t)**2))
ax.set_xlabel('x')
ax.set_ylabel(r'Probability density  $|\psi|^2$')
ax.set_xlim(-30,30)
ax.legend()
title = ax.set_title('')
ax.axvspan( -a/2, a/2, color='grey', alpha=0.62, label='barrier')

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
    line_prob.set_data([], [])
    return line_prob, title

def update(i):
    prob = np.abs(Psi_t[i])**2
    line_prob.set_data(x, prob)
    title.set_text("Quantum Tunnelling effect")
    return line_prob, title

ani = FuncAnimation(fig, update, frames=len(times), init_func=init,
                    blit=False, interval=40)
ani.save("1DTunnelling.gif", writer="pillow", fps=44)

plt.show()


# -------- Spacetime Heatmap Diagram -------- 
plt.figure()
plt.imshow(np.abs(Psi_t)**2, extent=[x.min(), x.max(), times[-1], times[0]],
           aspect='auto', cmap='magma')
plt.colorbar(label=r'$|\psi|^2$')
plt.xlabel('x')
plt.ylabel('time')
plt.title('Wavepacket Tunnelling spacetime heatmap')
plt.show()

