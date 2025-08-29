import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


#------- Time Dependent Schrodinger equation in 1 Dimension ----------
def crank_nicolson_tdse_1D(x, Vx, psi0, dt, nsteps, ħ=1.0, m=1.0, store_every=1):
    """
    Solve i ħ dψ/dt = [ -ħ²/(2m) d²/dx² + V(x) ] ψ  with Crank-Nicolson.
    Uses Dirichlet boundaries (ψ=0 at edges).
    
    Parameters
    ----------
    x : (N,) array
        Spatial grid (uniform spacing required).
    Vx : (N,) array
        Potential evaluated on x.
    psi0 : (N,) complex array
        Initial wavefunction at t=0.
    dt : float
        Time step.
    nsteps : int
        Number of time steps to evolve.
    ħ, m : float
        Planck's reduced constant and mass (set both =1.0 for dimensionless HO).
    store_every : int
        Store every k-th step to reduce memory (default 1 = store all).
        
    Returns
    -------
    Psi_t : (n_saved, N) complex array
        Wavefunction snapshots over time.
    times  : (n_saved,) float array
        Times corresponding to saved snapshots.
    """
    N = x.size
    dx = x[1] - x[0]
    main = np.full(N, -2.0)
    off  = np.ones(N-1)
    lap  = (sp.diags([off, main, off], [-1, 0, 1], shape=(N, N)) / dx**2).tocsr()

    T    = -(ħ**2)/(2.0*m) * lap
    V  = sp.diags(Vx, 0).tocsr()
    H = T + V                                                 # Hamiltonian = Kinetic + Potential 

    I = sp.identity(N, format='csr')                         # Crank–Nicolson matrices
    a = 1j * dt / (2.0 * ħ)                          
    A = (I + a * H).tocsr()
    B = (I - a * H).tocsr()

    solver = spla.splu(A.tocsc())                             # Pre-factorizing a

    psi = psi0.astype(np.complex128)                          # Time stepping
    psi_t = []
    time   = []
    t = 0.0

    for n in range(nsteps+1):
        if n % store_every == 0:
            psi_t.append(psi.copy())
            time.append(t)

        if n == nsteps:
            break

        rhs = B @ psi
        psi = solver.solve(rhs)

        t += dt

    return np.array(psi_t), np.array(time)


#------- Time Dependent Schrodinger equation in 2 Dimension ----------

def crank_nicolson_tdse_2d(x, y, Vxy, psi0, dt, steps, *, hbar = 1.0, mass = 1.0, save_every = 10, renorm = True):
    """
    Crank-Nicolson in 2D: iħ ψ_t = [ -ħ²/(2m) ∇² + V(x,y) ] ψ
      x, y      : 1D grids (uniform)
      Vxy       : (Nx,Ny) array (can be complex if CAP used)
      psi0      : initial state (Nx,Ny)
      dt, steps : time step and number of steps
      save_every: store every k-th frame
    Returns:
      frames : (n_frames, Nx, Ny) complex array
      times  : (n_frames,) float array
    """
    Nx, Ny = len(x), len(y)
    dx,dy = (x[1]-x[0]),(y[1]-y[0])
    
    mainx = np.full(Nx, -2.0)
    offx  = np.ones(Nx-1)
    Lx1  = (sp.diags([offx, mainx, offx], [-1, 0, 1], shape=(Nx, Nx)) / dx**2).tocsr()

    mainy = np.full(Ny, -2.0)
    offy  = np.ones(Ny-1)
    Ly1  = (sp.diags([offy, mainy, offy], [-1, 0, 1], shape=(Ny, Ny)) / dy**2).tocsr()
    Lap = sp.kronsum(Lx1, Ly1).tocsr()

    T   = -(hbar**2)/(2.0*mass) * Lap
    V = sp.diags(Vxy.ravel(), 0)
    H   = (T + V).tocsr()

    I = sp.identity(Nx*Ny, format='csr')
    a = 1j * dt / (2.0 * hbar)
    A = (I + a*H).tocsc()
    B = (I - a*H).tocsr()
    solver = spla.splu(A)  

    norm = np.trapezoid(np.trapezoid(np.abs(psi0)**2, y), x)
    psi_norm = psi0/ np.sqrt(norm)
    psi = psi_norm.ravel().astype(np.complex128)

    psi_t = []
    times = []
    for n in range(steps+1):
        if n % save_every == 0:
            psi_t.append(psi.reshape(Nx,Ny).copy())
            times.append(n*dt)
        if n == steps:
            break
        rhs = B @ psi
        psi = solver.solve(rhs)
        if renorm:
            psi2d = psi.reshape(Nx,Ny)
            norm = np.trapezoid(np.trapezoid(np.abs(psi2d)**2, y), x)
            psi2d_norm = psi2d/np.sqrt(norm)
            psi = psi2d_norm.ravel()

    return np.array(psi_t), np.array(times)


#------- Grid and Potential setup in 2 Dimension ----------

def ho2d_grids(Nx=300, Ny=300, Lx=12.0, Ly=12.0):
    x = np.linspace(-Lx/2, Lx/2, Nx)
    y = np.linspace(-Ly/2, Ly/2, Ny)
    return x, y

def ho2d_potential(x, y, omega=1.0):
    X, Y = np.meshgrid(x, y, indexing='ij')
    V = 0.5 * omega**2 * (X**2 + Y**2)
    return V

def gaussian2d_packet(x, y, x0=-2.5, y0=0.0, sigx=0.8, sigy=0.8, kx=2.0, ky=0.0):
    X, Y = np.meshgrid(x, y, indexing='ij')
    psi = (1/np.sqrt(np.pi*sigx*sigy)) * np.exp(-((X-x0)**2/(2*sigx**2) + (Y-y0)**2/(2*sigy**2))) \
          * np.exp(1j*(kx*X + ky*Y))
    return psi


def double_slit_barrier(x, y, x0=0.0, V0=1e4, thickness=0.2, slit_centers=(-0.5,0.5), slit_width=0.2):  
    """
    Returns V_b(x,y): a vertical wall at x=x0 with two rectangular slits.
    """
    X, Y = np.meshgrid(x, y, indexing='ij')
    Vb = np.zeros_like(X)
    mask_wall = np.abs(X - x0) <= (thickness * 0.5)
    Vb[mask_wall] = V0

    for yc in slit_centers:
        mask_slit = mask_wall & (np.abs(Y - yc) <= (slit_width * 0.5))
        Vb[mask_slit] = 0.0

    return Vb

def sinai_billiard_potential(x, y, V0=1e6, circle_radius=2.0, circle_center=(0.0,0.0), wall_cells=3.0):
    """
    """
    X, Y = np.meshgrid(x, y, indexing='ij')
    V = np.zeros_like(X, dtype=float)

    cx, cy = circle_center
    mask_circle = (X-cx)**2 + (Y-cy)**2 <= circle_radius**2
    V[mask_circle] = V0

    Nx, Ny = X.shape
    wc = max(1,int(wall_cells)) 

    V[:wc, :] = V0
    V[-wc:, :] = V0
    V[:, :wc] = V0
    V[:, -wc:] = V0

    return V

#------- Time Dependent Schrodinger equation in 3 Dimension ----------

def split_operator_3d(psi, V, K2, dt, nsteps, hbar=1.0, m=1.0, time_dependent_V=None, save_every=0, t0=0.0):
    phase_V = np.exp(-1j* V *dt/2*hbar)
    phase_T = None
    phase_T = np.exp(-1j*hbar*(K2*dt)/2*m)
     
    frames, times = [], []
    t = t0

    for n in range(nsteps):
        if time_dependent_V is not None:
            
            t_mid = (n + 0.5) * dt
            V_mid = time_dependent_V(t_mid)
            phase_V = np.exp(-1j * V_mid * (dt/(2*hbar)))
        psi *= phase_V

        psi_k = np.fft.fftn(psi,axes=(0,1,2))
        psi_k *= phase_T
        psi = np.fft.ifftn(psi_k,axes=(0,1,2))

        psi *= phase_V

        t += dt

        if save_every and ((n+1) % save_every == 0):
            frames.append(psi)
            times.append(t)

    return psi, np.array(frames), np.array(times)

# ---------- basics ----------
def V_ho_2d(X, Y, wx=1.0, wy=1.0):
    """Anisotropic harmonic oscillator."""
    return 0.5*(wx**2 * X**2 + wy**2 * Y**2)

def V_linear_tilt(X, Y, F=0.1, angle=0.0):
    """Uniform force (Stark-like tilt) in direction 'angle' (rad)."""
    return F*(np.cos(angle)*X + np.sin(angle)*Y)

# ---------- double wells / barriers ----------
def V_double_well_x(X, Y, a=0.02, b=2.0, wy=0.8):
    """Quartic double well along x + HO confinement in y."""
    return a*(X**2 - b**2)**2 + 0.5*wy**2 * Y**2

def V_double_barrier_dot(x, y, V0=6.0, width=0.6, sep=2.5, s=0.15, wy=0.6):
    """Two smooth barriers along x -> quantum dot between them."""
    X, Y = np.meshgrid(x, y, indexing='ij')
    def step(u): return 0.5*(1 + np.tanh(u/s))
    left  = step(X - (-(sep/2 - width/2))) - step(X - (-(sep/2 + width/2)))
    right = step(X - ( (sep/2 + width/2))) - step(X - ( (sep/2 - width/2)))
    return V0*(left + right) + 0.5*wy**2 * Y**2

# ---------- rings / “mexican hat” ----------
def V_ring_gaussian(x, y, V0=-8.0, r0=3.0, w=0.6):
    """Attractive Gaussian ring well at radius r0 (negative V0 binds)."""
    X, Y = np.meshgrid(x, y, indexing='ij')
    r = np.sqrt(X**2 + Y**2)
    return V0*np.exp(- (r - r0)**2 / (2*w**2))

def V_mexican_hat_2d(x, y, alpha=0.02, r0=3.0):
    """Quartic ring with minimum on a circle of radius r0."""
    X, Y = np.meshgrid(x, y, indexing='ij')
    r2 = X**2 + Y**2
    return alpha*(r2 - r0**2)**2

def V_circular_well(x, y, V0=-8.0, R=3.0, s=0.25):
    """Finite circular well (smooth edge)."""
    X, Y = np.meshgrid(x, y, indexing='ij')
    r = np.sqrt(X**2 + Y**2)
    inside = 0.5*(1 - np.tanh((r - R)/s))  # ≈1 inside, 0 outside
    return V0 * inside

def V_annulus_well(x, y, V0=-8.0, R1=2.0, R2=3.5, s=0.25):
    """Annular (ring) well between radii R1 and R2."""
    X, Y = np.meshgrid(x, y, indexing='ij')
    r = np.sqrt(X**2 + Y**2)
    inner = 0.5*(1 - np.tanh((r - R1)/s))
    outer = 0.5*(1 + np.tanh((r - R2)/s))
    return V0 * inner * outer

# ---------- lattices ----------
def V_lattice_square(x, y, V0=2.0, k=np.pi/2):
    """Square optical lattice (standing-wave sin^2)."""
    X, Y = np.meshgrid(x, y, indexing='ij')
    return V0*(np.sin(k*X)**2 + np.sin(k*Y)**2)

def V_lattice_triangular(x, y, V0=2.0, k=2*np.pi/3):
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
def V_speckle_2d(x, y, amp=1.0, corr=1.0, seed=0):
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








#------- Grid and Potential setup in 3 Dimension ----------

def grids3d(Nx=300, Ny=300, Nz=300, Lx=12.0, Ly=12.0, Lz=12.0):
    """
    Uniform 3D grid centered at 0. Returns x,y,z and spacings dx,dy,dz.
    """
    x = np.linspace(-Lx/2, Lx/2, Nx)
    y = np.linspace(-Ly/2, Ly/2, Ny)
    z = np.linspace(-Lz/2, Lz/2, Nz)
    dx, dy, dz = x[1]-x[0], y[1]-y[0], z[1]-z[0]

    return x, y, z, dx, dy, dz

def kgrids_3d(Nx, Ny, Nz, dx, dy, dz):
    """
    Fourier forward transform compatible grid to use in slit operator for 3d mesh grids. 
    """
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=dy)
    kz = 2*np.pi*np.fft.fftfreq(Nz, d=dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2 
    
    return kx, ky, kz, K2

def potential_3d(x, y, z, omega=1.0):
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    V = 0.5 * omega**2 * (X**2 + Y**2+ Z**2)
    return V

def gaussian_packet_3d(x, y, z, x0=-2.5, y0=0.0, z0 = 0.0, sigx=0.8, sigy=0.8, sigz=0.8, kx=2.0, ky=0.0, kz=0.0):
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    psi = (1/np.sqrt(np.pi*sigx*sigy*sigz)) * np.exp(-((X-x0)**2/(2*sigx**2) + (Y-y0)**2/(2*sigy**2) + (Z-z0)**2/(2*sigz**2))) \
          * np.exp(1j*(kx*X + ky*Y + kz*Z))
    dx, dy, dz = (x[1]-x[0]), (y[1]-y[0]), (z[1]-z[0])
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx * dy * dz)
    return psi / norm

def split_operator_3d_2(psi, V, K2, dt, nsteps, hbar=1.0, m=1.0,
                      time_dependent_V=None, save_every=0, t0=0.0):

    phase_V = np.exp(-1j * V * (dt/(2*hbar)))
    phase_T = np.exp(-1j * (hbar * K2 / (2*m)) * dt)

    frames, times = [], []
    t = t0

    for n in range(nsteps):
        if time_dependent_V is not None:
            t_mid = t + 0.5*dt
            V_mid = time_dependent_V(t_mid)
            phase_V = np.exp(-1j * V_mid * (dt/(2*hbar)))

        # V half-step
        psi *= phase_V

        # K full-step
        psi_k = np.fft.fftn(psi, axes=(0,1,2))
        psi_k *= phase_T
        psi = np.fft.ifftn(psi_k, axes=(0,1,2))

        # V half-step
        psi *= phase_V

        t += dt
        if save_every and ((n+1) % save_every == 0):
            frames.append(psi.copy())
            times.append(t)

    return np.array(frames), np.array(times)
