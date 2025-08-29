"""
2D Harmonic Oscillator — Time-Dependent Schrödinger Equation (TDSE) 

This script simulates the quantum dynamics of a Gaussian wavepacket in a
2D harmonic oscillator potential using the Crank-Nicolson method for time
evolution. 
"""



import numpy as np
import sys, os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


current_dir = os.path.dirname(os.path.abspath(__file__))    
project_root = os.path.abspath(os.path.join(current_dir, ".."))  
sys.path.insert(0, project_root)

from Master.TDSE import ho2d_grids, ho2d_potential, crank_nicolson_tdse_2d, gaussian2d_packet


# ------- Grid & Potential Setup -------
Nx,Ny = 100,100
Lx,Ly = 12.0,12.0
x,y = ho2d_grids(Nx,Ny,Lx,Ly)

omega = 1.0
V = ho2d_potential(x,y,omega)

psi0 = gaussian2d_packet(x, y, x0=-2.5, y0=0.0, sigx=1.0, sigy=1.0, kx=2.0, ky=1.0)


# ------- Evolve -------
dt = 0.005
steps = 100
save_every = 10
frames, times = crank_nicolson_tdse_2d(x, y, V, psi0, dt, steps, hbar=1.0, mass=1.0, save_every=save_every)

densities = [np.abs(f)**2 for f in frames]
prob0 = np.abs(frames[0])**2

# ------- animate |psi|^2 --------
fig, ax = plt.subplots(figsize=(6,5))
prob0 = np.abs(frames[0])**2
im = ax.imshow(prob0.T, extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', cmap='gnuplot', aspect='equal', vmin=0, vmax=prob0.max())
cb = plt.colorbar(im, ax=ax, label=r'$|\psi(x,y,t)|^2$')
title = ax.set_title("t = 0.000")
ax.set_xlabel('x'); ax.set_ylabel('y')

def update(i):
    prob = np.abs(frames[i])**2
    im.set_data(prob.T)
    # im.set_clim(0, prob.max())  # autoscale if you prefer
    title.set_text(f"t = {times[i]:.3f}")
    return (im,)

ani = FuncAnimation(fig, update, frames=len(times), interval=2, blit=False)
plt.show()