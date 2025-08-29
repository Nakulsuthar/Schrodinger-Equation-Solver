
"""
1D Harmonic Oscillator - Time Independent Schrodingers Equation (TISE)

This script solves the TISE for a quantum harmonic oscillator by diagonalising 
the hamiltonian using the method in the hamiltonian file under the function solve.
The eigenvalues and the eigenvectors are calculated and the 10 eigenstates are plotted 
alongside the probability density of each state. 

"""

import numpy as np
import sys, os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


current_dir = os.path.dirname(os.path.abspath(__file__))    
project_root = os.path.abspath(os.path.join(current_dir, ".."))  
sys.path.insert(0, project_root)

from Master.hamiltonian import hamiltonian
from Master.constants import eV,Å 


# --------------- Potential and Hamiltonian Setup ----------------
def Harmonic_Oscillator(x):
    k = 100* eV / Å**2
    return 0.5 * k * x **2 

H = hamiltonian(
    N = 5120, 
    extent = 20*Å,
    mass = 1.0,
    spatial_ndim = 1,
    potential = Harmonic_Oscillator,
    potential_type = "grid",
)



# ---------------- EigenEnergy/EigenState Solver -------------------

energies, states = H.solve(max_states=30)


# -------------------- SubPlots Setup ---------------------------

x = np.linspace(-H.extent[0]/2, H.extent[0]/2, H.N[0])


n_init = 0


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
plt.subplots_adjust(bottom=0.2)  

wave_line, = ax1.plot(x, states[:, n_init], color="Cyan")
ax1.set_ylabel(r"Wavefunction $\psi(x)$")
ax1.set_title(f"Harmonic Oscillator: State n={n_init}")
ax1.set_ylim(-0.08,0.08)
ax1.grid(True)

prob_line, = ax2.plot(x, states[:, n_init]**2, color="yellow")
V_diag = np.array(H.V.diagonal())
V_scaled = V_diag / np.max(V_diag) * np.max(states[:, n_init]**2)
potential_line, = ax2.plot(x, V_scaled, 'k--', label="Scaled $V(x)$",color='white')
ax2.set_xlabel(r"Position $x$ [Å]")
ax2.set_ylabel(r"Probability Density $|\psi(x)|^2$")
ax2.grid(True)
ax2.legend(frameon=False)


fig.patch.set_facecolor('black')
ax1.set_facecolor('black')
ax2.set_facecolor('black')
ax1.tick_params(colors='white')
ax2.tick_params(colors='white')
ax1.xaxis.label.set_color('white')
ax2.xaxis.label.set_color('white')
ax1.yaxis.label.set_color('white')
ax2.yaxis.label.set_color('white')
ax1.title.set_color('white')
ax2.title.set_color('white')
ax1.grid(False)
ax2.grid(False)
for spine in ax1.spines.values():
    spine.set_edgecolor('white')
for spine in ax2.spines.values():
    spine.set_edgecolor('white')
for text in ax1.legend().get_texts():
    text.set_color('white')
for text in ax2.legend().get_texts():
    text.set_color('white')

# ------------------ Slider Setup --------------------

ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 'State n', 0, states.shape[1]-1, valinit=n_init, valstep=1)

def update(val):
    n = int(slider.val)
    wave_line.set_ydata(states[:, n])
    prob_line.set_ydata(states[:, n]**2)

    scaled_potential = V_diag / np.max(V_diag) * np.max(states[:, n]**2)
    potential_line.set_ydata(scaled_potential)

    ax1.set_title(f"1D Harmonic Oscillator: State n={n}")
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()


