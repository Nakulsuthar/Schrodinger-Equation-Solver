import numpy as np
import sys, os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tqdm import tqdm


current_dir = os.path.dirname(os.path.abspath(__file__))    
project_root = os.path.abspath(os.path.join(current_dir, ".."))  
sys.path.insert(0, project_root)

from Master.hamiltonian import hamiltonian
from Master.constants import eV,Å,hbar, T

# ------------ ------------

def harmonic_magentic_interaction(x, y, Nx, Ny):
    kx = 2e-6
    ky = 2e-6 
    Harmonic_interaction = 0.5*kx*x**2 + 0.5*ky*y**2


    dx, dy = x[1]-x[0], y[1]-y[0]
    px = hbar * 2*np.pi*np.fft.fftfreq(Nx, d=dx)
    py = hbar * 2*np.pi*np.fft.fftfreq(Ny, d=dy)
    Bz = 30 * T
    B_dot_L = Bz * (-px @ y + py @ x)
    
    paramag_term = -0.5 * (B_dot_L)

    d = 0.125 
    diamag_term = d*Bz**2 * (x**2 + y**2)

    Magnetic_interaction = diamag_term + paramag_term

    return Harmonic_interaction + Magnetic_interaction

def V_HO_plus_B_diamagnetic(X, Y, m=1.0, q=-1.0, B=0.0, wx=1.0, wy=1.0):
    """Pure scalar potential: 2D HO + uniform B (diamagnetic A^2 only)."""
    V_ho  = 0.5*m*(wx**2*X**2 + wy**2*Y**2)
    V_dia = (q**2 * B**2)/(8.0*m) * (X**2 + Y**2)
    return V_ho + V_dia


H = hamiltonian(
    N = (258,258) , 
    extent = (20*Å,20*Å),
    mass = 1.0,
    spatial_ndim = 2,
    potential = lambda X, Y: V_HO_plus_B_diamagnetic(X, Y, m=1.0, q=-1.0, B=0.0, wx=1.0, wy=1.0) ,
    potential_type = "grid",
)


# =============================================

energies, states = H.solve(max_states=10)

# =============================================


n_states = states.shape[1]
psi_init = 0 
Nx,Ny = H.N if isinstance(H.N, (tuple, list)) else (H.N, H.N)
Lx, Ly = H.extent if isinstance(H.extent, (tuple, list)) else (H.extent, H.extent)

psi_plot = states[:,psi_init].reshape(H.N)
prob_plot = np.abs(psi_plot)**2

fig, ax = plt.subplots(figsize=(8,6))
plt.subplots_adjust(left=0.08, right=0.92, bottom=0.18, top=0.92)
im = ax.imshow(prob_plot.T, extent=[-Lx, Lx, -Ly, Ly],
               origin='lower', aspect='equal', cmap='magma')
cb = plt.colorbar(im, ax=ax, label=r'$|\psi(x,y)|^2$')
title = ax.set_title(f"State n={psi_init}")
ax.set_xlabel("x"); ax.set_ylabel("y")



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