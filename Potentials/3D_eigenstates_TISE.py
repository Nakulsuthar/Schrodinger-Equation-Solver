import os, sys
import numpy as np
import pyvista as pv
import time
import imageio

current_dir = os.path.dirname(os.path.abspath(__file__))    
project_root = os.path.abspath(os.path.join(current_dir, ".."))  
sys.path.insert(0, project_root)

from Master.hamiltonian import hamiltonian

# ----------- Potential and Hamiltonian setup -------------
def V_ho(x, y=None, z=None, omega=1.0, m=1.0):
    return 0.5*m*omega**2 * (x**2 + y**2 + z**2)

H = hamiltonian(N=(60,60,60), extent=(10.0,10.0,10.0), mass=1.0, spatial_ndim=3, potential=lambda X, Y, Z: V_ho(X, Y, Z), potential_type="grid")

# ------------- Solve for eigenstates -----------------
import time
start = time.time()
E, Psi = H.solve(max_states=10)
print(f"Solve time: {time.time() - start:.2f} seconds")



# Create the grid
x, y, z = [np.linspace(-L/2, L/2, n) for L, n in zip(H.extent, H.N)]
grid = pv.ImageData()
grid.dimensions = np.array(H.N) + 1
grid.origin = (-H.extent[0]/2, -H.extent[1]/2, -H.extent[2]/2)
grid.spacing = (H.extent[0]/H.N[0], H.extent[1]/H.N[1], H.extent[2]/H.N[2])

# Helper function
def psi_to_rho(i):
    psi = Psi[:, i].reshape(H.N, order="C")
    return (np.abs(psi)**2).astype(np.float32)

# PyVista plotter
plotter = pv.Plotter(off_screen=True, window_size=(600,600))
plotter.set_background("Black")
plotter.add_axes(color="white")

frames = []
n_states = 10    # cycle through first 10 states
n_frames = 120    # total frames for GIF
orbit_steps = 360 # how many degrees the camera orbits

for frame in range(n_frames):
    state = (frame // (n_frames // n_states)) % n_states
    rho = psi_to_rho(state)
    grid.point_data.clear()
    grid["rho"] = rho.flatten(order="C")

    plotter.clear()
    plotter.add_volume(grid, scalars="rho", cmap="gnuplot", opacity="sigmoid_5")

    # Orbit camera
    azimuth = (frame / n_frames) * orbit_steps
    plotter.camera.azimuth = azimuth
    plotter.camera.elevation = 20  # slight tilt

    img = plotter.screenshot(return_img=True)
    frames.append(img)

# Save as GIF
imageio.mimsave("3D_eigenstates1.gif", frames, fps=10)
print("Saved 3D_eigenstates.gif")
