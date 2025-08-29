import os, sys
import numpy as np
import pyvista as pv
import time

current_dir = os.path.dirname(os.path.abspath(__file__))    
project_root = os.path.abspath(os.path.join(current_dir, ".."))  
sys.path.insert(0, project_root)

from Master.hamiltonian import hamiltonian

# ----------- Potential and Hamiltonian setup -------------
def V_ho(x, y=None, z=None, omega=1.0, m=1.0):
    return 0.5*m*omega**2 * (x**2 + y**2 + z**2)

def V_double_well(X, Y, Z, a=0.02, b=2.0, omega_perp=1.0):
    Vx = a*(X**2 - b**2)**2
    Vperp = 0.5*omega_perp**2*(Y**2 + Z**2)
    return Vx + Vperp                                              

def V_mexican_hat(X, Y, Z, alpha=0.02, r0=3.0, wz=1.0):
    r2 = X**2 + Y**2
    return alpha*(r2 - r0**2)**2 + 0.5*wz**2*Z**2                     

def V_optical_lattice(X, Y, Z, V0=2.0, kL=np.pi/2):
    return V0*(np.sin(kL*X)**2 + np.sin(kL*Y)**2 + np.sin(kL*Z)**2)     

def V_hydrogen_soft(X, Y, Z, Zc=1.0, eps=0.4):
    r2 = X**2 + Y**2 + Z**2
    N = (128,128,128)
    extent = (40.0, 40.0, 40.0)
    dx = extent[0]/(N[0]-1)
    eps1 = eps*dx
    return -Zc / np.sqrt(r2 + eps1**2)



H = hamiltonian(N=(60,60,60), extent=(10.0,10.0,10.0), mass=1.0, spatial_ndim=3, potential=lambda X, Y, Z: V_optical_lattice(X, Y, Z), potential_type="grid")

# ------------- Solve for eigenstates -----------------
import time
start = time.time()
E, Psi = H.solve(max_states=10)
print(f"Solve time: {time.time() - start:.2f} seconds")


# -------------- Pyvista 3D Plot setup -----------------
axes = [np.linspace(-L/2, L/2, n) for L, n in zip(H.extent, H.N)] 

def psi_to_rho(i):
    psi = Psi[:, i].reshape(H.N, order="C")
    return (np.abs(psi)**2).astype(np.float32)

x, y, z = axes
rho0 = psi_to_rho(0)

grid = pv.ImageData()
grid.dimensions = rho0.shape          
grid.origin = (x[0], y[0], z[0])
grid.spacing = (x[1]-x[0], y[1]-y[0], z[1]-z[0])
grid["rho"] = rho0.ravel(order="F").astype(np.float32)


p = pv.Plotter()
vol = p.add_volume(grid, scalars="rho", opacity="sigmoid_5", cmap="inferno")
p.set_background("Black")
p.add_axes(color="white")
k = Psi.shape[1]

# --------------- Slider tool -------------
def on_slide(val):
    i = int(round(val))
    rho = psi_to_rho(i).ravel(order="F").astype(np.float32)
    grid.point_data["rho"][:] = rho       
    grid.point_data.Modified()              
    grid.SetText(2, f"state {i}   E={E[i]:.4f}")
    p.render()

p.add_slider_widget(
    callback=on_slide,
    rng=[0, k-1],
    value=0,
    title="",
    fmt="%.0f",
    style="modern",
)

# --------------- Automatic camera panning tool ---------------
center = np.array(grid.center)
L = grid.length
R = 1.3 * L                 
elev = np.deg2rad(20)           

def orbit_once():
    n = 240                       
    for i in range(n):
        phi = 2*np.pi * (i/n)
        pos = center + R * np.array([np.cos(phi)*np.cos(elev),
                                     np.sin(phi)*np.cos(elev),
                                     np.sin(elev)])
        p.camera_position = (pos.tolist(), center.tolist(), (0,0,1))
        p.render()

p.add_key_event("o", orbit_once)  

p.show()


"""
import imageio

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
    plotter.camera.elevation = 0  # slight tilt

    img = plotter.screenshot(return_img=True)
    frames.append(img)

# Save as GIF
imageio.mimsave("3D_eigenstates_optical_lattice.gif", frames, fps=10)
print("Saved 3D_eigenstates_tophat.gif")
"""
