import numpy as np
import sys, os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


current_dir = os.path.dirname(os.path.abspath(__file__))    
project_root = os.path.abspath(os.path.join(current_dir, ".."))  
sys.path.insert(0, project_root)

from Master.TDSE import ho2d_grids, crank_nicolson_tdse_2d, gaussian2d_packet,sinai_billiard_potential

# ------- Grid & Potential Setup -------
Nx, Ny = 200, 200          
Lx, Ly = 30.0, 20.0        
x, y = ho2d_grids(Nx, Ny, Lx, Ly)

# ------- Potential: Sinai Billiards -------

Potential = sinai_billiard_potential(
    x, y, 
    V0 = 1e6,
    circle_radius=  2.0,
    circle_center=(4.0,0.0),
    wall_cells=3
)


# ------- Initial Gaussian Packet -------
psi0 = gaussian2d_packet(
    x, y,
    x0=10.0,                
    y0=0.0,
    sigx=1.5,               
    sigy=1.5,               
    kx=7.0,                
    ky=2.0
)

# ------- Time evolution -------
dt = 0.009              
steps = 1000           
save_every = 10
frames, times = crank_nicolson_tdse_2d(
    x, y, Potential, psi0, dt, steps, hbar=1.0, mass=1.0, save_every=save_every
)



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
    title.set_text(f"t = {times[i]:.3f}")
    return (im,)

ani = FuncAnimation(fig, update, frames=len(times), interval=2, blit=False)
ani.save("SinaiPotential2.gif", writer="pillow", fps=60)
plt.show()

