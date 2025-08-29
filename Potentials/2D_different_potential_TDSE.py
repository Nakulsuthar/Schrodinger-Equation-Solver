"""
2D Different Potential - Time Dependent Schrodingers Equation 

This script solves the TDSE under different types of interaction 
potential given as a choice when run, the type of particle is also 
given as a choice between a single guassian and a superposition of 
two moving packets. An animation is then produced from the selected choices.
"""
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


current_dir = os.path.dirname(os.path.abspath(__file__))    
project_root = os.path.abspath(os.path.join(current_dir, ".."))  
sys.path.insert(0, project_root)

from Master.TDSE import ho2d_grids, crank_nicolson_tdse_2d, gaussian2d_packet


# ---------- double barriers ----------

def V_double_barrier_dot(X, Y, V0=6.0, width=0.6, sep=2.5, s=0.15, wy=0.6):
    """Two smooth barriers along x -> quantum dot between them."""
    X, Y = np.meshgrid(x, y, indexing='ij')
    def step(u): return 0.5*(1 + np.tanh(u/s))
    left  = step(X - (-(sep/2 - width/2))) - step(X - (-(sep/2 + width/2)))
    right = step(X - ( (sep/2 + width/2))) - step(X - ( (sep/2 - width/2)))
    return V0*(left + right) + 0.5*wy**2 * Y**2

# ---------- rings / “mexican hat” ----------
def V_ring_gaussian(X, Y, V0=-8.0, r0=3.0, w=0.6):
    """Attractive Gaussian ring well at radius r0 (negative V0 binds)."""
    X, Y = np.meshgrid(x, y, indexing='ij')
    r = np.sqrt(X**2 + Y**2)
    return V0*np.exp(- (r - r0)**2 / (2*w**2))

def V_mexican_hat_2d(X, Y, alpha=0.02, r0=3.0):
    """Quartic ring with minimum on a circle of radius r0."""
    X, Y = np.meshgrid(x, y, indexing='ij')
    r2 = X**2 + Y**2
    return alpha*(r2 - r0**2)**2

def V_circular_well(X, Y, V0=-8.0, R=3.0, s=0.25):
    """Finite circular well (smooth edge)."""
    X, Y = np.meshgrid(x, y, indexing='ij')
    r = np.sqrt(X**2 + Y**2)
    inside = 0.5*(1 - np.tanh((r - R)/s))  # ≈1 inside, 0 outside
    return V0 * inside

def V_annulus_well(X, Y, V0=-8.0, R1=2.0, R2=3.5, s=0.25):
    """Annular (ring) well between radii R1 and R2."""
    X, Y = np.meshgrid(x, y, indexing='ij')
    r = np.sqrt(X**2 + Y**2)
    inner = 0.5*(1 - np.tanh((r - R1)/s))
    outer = 0.5*(1 + np.tanh((r - R2)/s))
    return V0 * inner * outer

# ---------- lattices ----------
def V_lattice_square(X, Y, V0=2.0, k=np.pi/2):
    """Square optical lattice (standing-wave sin^2)."""
    X, Y = np.meshgrid(x, y, indexing='ij')
    return V0*(np.sin(k*X)**2 + np.sin(k*Y)**2)

def V_lattice_triangular(X, Y, V0=2.0, k=2*np.pi/3):
    """Triangular lattice from 3 beams (120°)."""
    X, Y = np.meshgrid(x, y, indexing='ij')
    c60, s60 = 0.5, np.sqrt(3)/2
    u1 = k*X
    u2 = k*(-c60*X + s60*Y)
    u3 = k*(-c60*X - s60*Y)
    # shift so min ~0, positive overall
    raw = (np.cos(u1) + np.cos(u2) + np.cos(u3))
    return V0*(raw - raw.min())

# ---------- disorder / speckle ----------
def V_speckle_2d(X, Y, amp=1.0, corr=1.0, seed=0):
    """Gaussian random field with correlation length ~corr (FFT filter)."""
    X, Y = np.meshgrid(x, y, indexing='ij')
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=X.shape)
    Fx = np.fft.fftfreq(X.shape[0], d=(X[1,0]-X[0,0]))
    Fy = np.fft.fftfreq(X.shape[1], d=(Y[0,1]-Y[0,0]))
    KX, KY = np.meshgrid(Fx, Fy, indexing="ij")
    filt = np.exp(-0.5*( (2*np.pi*corr)**2 ) * (KX**2 + KY**2))
    smooth = np.real(np.fft.ifft2(np.fft.fft2(noise)*filt))
    smooth -= smooth.mean(); smooth /= (smooth.std() + 1e-12)
    return amp*smooth

    
Nx,Ny = 200,200
Lx,Ly = 14.0,12.0
x,y = ho2d_grids(Nx,Ny,Lx,Ly)

X, Y = np.meshgrid(x, y, indexing="ij")

print("Choose an interaction potential")
print("1 = Double Barrier Dot")
print("2 = Ring Gaussian")
print("3 = Mexican Hat")
print("4 = Circular Well")
print("5 = Annulus Well")
print("6 = Lattice Squares")
print("7 = Lattice Triangular")
print("8 = Speckle")
choice = input("Enter 1, 2, 3, 4, 5, 6, 7, or 8: ").strip()

if choice == "1":
    V = V_double_barrier_dot(X, Y, V0=6.0, width=0.6, sep=2.5, s=0.15, wy=0.6)
elif choice == "2":
    V = V_ring_gaussian(X, Y, V0=-8.0, r0=3.0, w=0.6)
elif choice == "3":
    V = V_mexican_hat_2d(X, Y, alpha=0.02, r0=3.0)
elif choice == "4":
    V = V_circular_well(X, Y, V0=-8.0, R=3.0, s=0.25)
elif choice == "5":
    V = V_annulus_well(X, Y, V0=-8.0, R1=2.0, R2=3.5, s=0.25)
elif choice == "6":
    V = V_lattice_square(X, Y, V0=2.0, k=np.pi/2)
elif choice == "7":
    V = V_lattice_triangular(X, Y, V0=2.0, k=2*np.pi/3)
elif choice == "8":
    V = V_speckle_2d(X, Y, amp=1.0, corr=1.0, seed=0)
else:
    raise SystemExit(f"Invalid choice: {choice}")


psi0 = gaussian2d_packet(x, y, x0=-5.5, y0=0.0, sigx=1.0, sigy=1.0, kx=2.0, ky=1.0)


def normalize(psi, x, y):
    dx, dy = (x[1]-x[0]), (y[1]-y[0])
    nrm = np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)
    return psi / (nrm + 1e-300)

psi1 = gaussian2d_packet(x, y, x0=-2.5, y0=0.0, sigx=1.0, sigy=1.0, kx=2.0, ky=1.0)
psi2 = gaussian2d_packet(x, y, x0=+2.5, y0=0.0, sigx=1.0, sigy=1.0, kx=-2.0, ky=1.0)

a1, a2 = 1.0, 0.8*np.exp(1j*np.pi/4) 
psi3 = normalize(a1*psi1 + a2*psi2, x, y)

print("Choose what kind of particle do you want in your interaction potential")
print("1 = Single Guassian wave packet")
print("2 = Superposition of two Guassian wave packets")

choice2 = input("Enter 1, or 2: ").strip()

if choice2 == "1":
    psi = psi0
elif choice2 =="2":
    psi = psi3
else:
    raise SystemExit(f"Invalid choice: {choice2}")


# ------- Evolve -------
dt = 0.005
steps = 2000
save_every = 10
frames, times = crank_nicolson_tdse_2d(x, y, V, psi, dt, steps, hbar=1.0, mass=1.0, save_every=save_every)

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
    title.set_text(f"t = {times[i]:.3f}")
    return (im,)

ani = FuncAnimation(fig, update, frames=len(times), interval=2, blit=False)
ani.save(f"2Dchoice{choice}44fpsV3 .gif", writer="pillow", fps=44)
plt.show()