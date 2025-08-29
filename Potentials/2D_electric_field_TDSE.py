"""
====
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
Nx, Ny = 160, 160
Lx, Ly = 16.0, 16.0
x, y   = ho2d_grids(Nx, Ny, Lx, Ly)

omega  = 1.0
X, Y   = np.meshgrid(x, y, indexing="ij")


# choose sign of charge (electron-like -> q = -1)
q = -2.0
Ex, Ey = 0.2, 0.2   # example field in code units

def V_uniform_E_2d(X, Y, Ex, Ey, q):
    return -q * (Ex*X + Ey*Y)

# (optional) base potential, e.g. 2D HO
def V_ho_2d(X, Y, omega=1.0):
    return 0.5 * omega**2 * (X**2 + Y**2)

# (recommended) absorbing edges to avoid reflections
def cap_2d(X, Y, Lx, Ly, w_frac=0.12, eta=2.0):
    wx, wy = w_frac*Lx, w_frac*Ly
    def edge(s, L, w):
        d = np.maximum(0.0, np.abs(s) - (L/2 - w))
        return (d/w)**2
    return -1j*eta * (edge(X, Lx, wx) + edge(Y, Ly, wy))

Vxy = V_ho_2d(X, Y, omega=1.0) + V_uniform_E_2d(X, Y, Ex, Ey, q) + cap_2d(X,Y,Lx,Ly)


# ---- Initial packets: choose radius R and match ky = omega*R ----
R     = 2.5           
x0,y0 = -R, 0.0
sig   = 1.0           
kx    = 0.0
ky    = omega * R     

psi0  = gaussian2d_packet(x, y, x0=x0, y0=y0, sigx=sig, sigy=sig, kx=kx, ky=ky)


R     = 2.5           
x0,y0 = +R, 0.0
sig   = 1.0          
kx    = 0.0
ky    = omega * R     

psi1  = gaussian2d_packet(x, y, x0=x0, y0=y0, sigx=sig, sigy=sig, kx=kx, ky=ky)


def normalize(psi, x, y):
    dx, dy = (x[1]-x[0]), (y[1]-y[0])
    nrm = np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)
    return psi / (nrm + 1e-300)

psi2 = normalize(psi0 + psi1, x, y)

# ------- Evolve (one full period T=2π/omega ≈ 6.283) -------
dt    = 0.004
steps = 4000         
save_every = 10

frames, times = crank_nicolson_tdse_2d(
    x, y, Vxy, psi0, dt, steps,
    hbar=1.0, mass=1.0, save_every=save_every
)

densities = [np.abs(f)**2 for f in frames]
prob0 = np.abs(frames[0])**2

# ------- animate |psi|^2 --------
fig, ax = plt.subplots(figsize=(6,5))
prob0 = np.abs(frames[0])**2
im = ax.imshow(prob0.T, extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', cmap='magma', aspect='equal', vmin=0, vmax=prob0.max())
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
ani.save("ElectricField2DV2.gif", writer='pillow', fps=44)
plt.show()