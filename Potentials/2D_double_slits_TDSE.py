"""
2D Double Slit - Time Dependent Schrodingers Equation (TDSE)

This script solves the same schrodinger equation as did in 2D Harmonic 
Oscillator TDSE, however the interaction potential here is a hard wall 
with two narrow slits carved. This potential portrays the classic 
double slit experiment. 
"""


import numpy as np
import sys, os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


current_dir = os.path.dirname(os.path.abspath(__file__))    
project_root = os.path.abspath(os.path.join(current_dir, ".."))  
sys.path.insert(0, project_root)

from Master.TDSE import ho2d_grids, crank_nicolson_tdse_2d, gaussian2d_packet,double_slit_barrier


# ----------- Grid & Potential Setup -----------
Nx, Ny = 300, 300         
Lx, Ly = 30.0, 20.0       
x, y = ho2d_grids(Nx, Ny, Lx, Ly)


double_slit = double_slit_barrier(
    x, y,
    x0=0.0,
    V0=1000.0,              
    thickness=0.3,
    slit_centers=(-1.0, 1.0),
    slit_width=0.5
)


# ------- Initial Gaussian Packet -------
psi0 = gaussian2d_packet(
    x, y,
    x0=-13.0,                
    y0=0.0,
    sigx=1.5,               
    sigy=1,            
    kx=8.0,              
    ky=0.0
)

# ------- Time evolution -------
dt = 0.002               
steps = 800  
save_every = 10
frames, times = crank_nicolson_tdse_2d(
    x, y, double_slit, psi0, dt, steps, hbar=1.0, mass=1.0, save_every=save_every
)

# ------- animate |psi|^2 --------
fig, ax = plt.subplots(figsize=(6,5))
prob0 = np.abs(frames[0])**2
im = ax.imshow(prob0.T, extent=[-15, 25, y.min(), y.max()],
               origin='lower', cmap='gnuplot', aspect='equal', vmin=0, vmax=0.1)
cb = plt.colorbar(im, ax=ax, label=r'$|\psi(x,y,t)|^2$')
title = ax.set_title("t = 0.000")
ax.set_xlabel('x'); ax.set_ylabel('y')

def update(i):
    prob = np.abs(frames[i])**2
    im.set_data(prob.T)
    title.set_text(f"t = {times[i]:.3f}")
    return (im,)

ani = FuncAnimation(fig, update, frames=len(times), interval=50, blit=False)
ani.save("DoubleSlit2.gif", writer="pillow", fps=44)
plt.show()
