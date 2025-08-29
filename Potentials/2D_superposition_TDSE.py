import numpy as np
import sys, os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


current_dir = os.path.dirname(os.path.abspath(__file__))    
project_root = os.path.abspath(os.path.join(current_dir, ".."))  
sys.path.insert(0, project_root)

from Master.TDSE import ho2d_grids, ho2d_potential, crank_nicolson_tdse_2d, gaussian2d_packet,double_slit_barrier


# ------- Grid & Potential Setup -------
Nx,Ny = 200,200
Lx,Ly = 12.0,12.0
x,y = ho2d_grids(Nx,Ny,Lx,Ly)

omega = 1.0
V = ho2d_potential(x,y,omega)

def normalize(psi, x, y):
    dx, dy = (x[1]-x[0]), (y[1]-y[0])
    nrm = np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)
    return psi / (nrm + 1e-300)

psi1 = gaussian2d_packet(x, y, x0=-5.0, y0=5.0, sigx=1.0, sigy=1.0, kx=2.0, ky=1.0)
psi2 = gaussian2d_packet(x, y, x0=5.0, y0=5.0, sigx=1.0, sigy=1.0, kx=2.0, ky=1.0)
psi3= gaussian2d_packet(x, y, x0=5.0, y0=-5.0, sigx=1.0, sigy=1.0, kx=9.0, ky=1.0)

#a1, a2 = 1.0, 0.8*np.exp(1j*np.pi/4) 
a1, a2, a3= 1.0, 1.0, 1.0
psi0 = normalize(a1*psi1 + a2*psi2 + a3*psi3, x, y)


# ------- Evolve -------
dt = 0.005
steps = 2000
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