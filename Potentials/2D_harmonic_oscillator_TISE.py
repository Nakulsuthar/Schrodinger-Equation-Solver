"""
2D Harmonic Oscillator - Time Independent Schrodinger Equation

This script solves the schrodingers equation in 2 dimensions using finite 
differences method built in the hamiltonian by diagonalizing it for the lowest
eigenvalues/eigenfunctions. These eigenfunctions(states) are then plotted to show
how the states look like under a quantum harmonic oscillator.
"""

import numpy as np
import sys, os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


current_dir = os.path.dirname(os.path.abspath(__file__))    
project_root = os.path.abspath(os.path.join(current_dir, ".."))  
sys.path.insert(0, project_root)

from Master.hamiltonian import hamiltonian
from Master.constants import eV,Å,hbar 

# ------------ Potential and Hamiltonian Setup --------------

def Harmonic_Oscillator(x,y):
    k = 100* eV / Å**2
    return (0.5 * k * (x**2 + y**2))

H = hamiltonian(
    N = (256,256) , 
    extent = (20*Å,20*Å),
    mass = 1.0,
    spatial_ndim = 2,
    potential = Harmonic_Oscillator,
    potential_type = "grid",
)

# ---------------- EigenEnergy/EigenState Solver -----------------

energies, states = H.solve(max_states=50)

# -------------------- Plot Setup ---------------------------

n_states = states.shape[1]
psi_init = 0 
Nx,Ny = H.N if isinstance(H.N, (tuple, list)) else (H.N, H.N)
Lx, Ly = H.extent if isinstance(H.extent, (tuple, list)) else (H.extent, H.extent)

psi_plot = states[:,psi_init].reshape(H.N)
prob_plot = np.abs(psi_plot)**2

fig, ax = plt.subplots(figsize=(8,6))
plt.subplots_adjust(left=0.08, right=0.92, bottom=0.18, top=0.92)
im = ax.imshow(prob_plot.T, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2],
               origin='lower', aspect='equal', cmap='magma')
cb = plt.colorbar(im, ax=ax, label=r'$|\psi(x,y)|^2$')
title = ax.set_title(f"State n={psi_init}")
ax.set_xlabel("x"); ax.set_ylabel("y")


# -------------------- Slider Tool ---------------------------

ax_slider = fig.add_axes([0.12, 0.08, 0.76, 0.035])
slider = Slider(ax_slider, 'State n',
                valmin=0, valmax=n_states-1,
                valinit=psi_init, valstep=1, valfmt='%0.0f')

def update(val):
    n = int(slider.val)
    psi = states[:,n].reshape(H.N)
    prob = np.abs(psi)**2
    im.set_data(prob.T)
    im.set_clim(vmin=0, vmax=prob_plot.max())
    title.set_text(f"State n={psi_init} for 2D Harmonic Oscillator")
    fig.canvas.draw_idle()


slider.on_changed(update)
plt.show()




