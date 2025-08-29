import numpy as np 
from scipy.sparse import diags,kron,identity
from scipy.sparse.linalg import eigsh
from Master.constants import * 

class hamiltonian:
    def __init__(self, N, extent, mass=1.0, spatial_ndim =1, potential = None, potential_type = "grid"):
        """
        """
        self.spatial_ndim = spatial_ndim                  #Dimensionality (1D,2D,3D)
        self.mass = mass
        self.potential_type = potential_type              #Potential function (Grid) or operator (Matrix) based

        #Automatic grid setup
        if isinstance(N, int):                            # same N for all dimensions 
            self.N = (N,) * spatial_ndim
        else:
            self.N = tuple(N)
        if isinstance(extent,(float,int)):
            self.extent = (extent,) * spatial_ndim
        else:
            self.extent = tuple(extent)

        self.dx = [L / n for L, n in zip(self.extent,self.N)]
        self.potential = potential
        self.hbar = hbar 

        #Matrices setup 
        self.T = self.build_kinetic()
        self.V = self.build_potential()
        self.H = self.T + self.V

    def build_kinetic(self):
        """
        function to build Kinetic energy (T) operator using the method of finite differences for second derivates
        """
        coeff = -(hbar ** 2) / (2 * self.mass)

        def one_d_kinetic(N, dx):
            diag = np.full(N, -2.0)                          # main diagonal (-2)
            off_diag = np.ones(N-1)                          # off diagonal (+1)
            return coeff / dx ** 2 * diags([off_diag,diag,off_diag], [-1,0,1])

        if self.spatial_ndim == 1:
            return one_d_kinetic(self.N[0], self.dx[0])
        elif self.spatial_ndim == 2:
            Tx = one_d_kinetic(self.N[0], self.dx[0])
            Ty = one_d_kinetic(self.N[1], self.dx[1])
            Ix = identity(self.N[0])
            Iy = identity(self.N[1])
            return kron(Tx, Iy) + kron(Ix, Ty)
        elif self.spatial_ndim == 3:
            Tx = one_d_kinetic(self.N[0],self.dx[0])
            Ty = one_d_kinetic(self.N[1],self.dx[1])
            Tz = one_d_kinetic(self.N[2],self.dx[2])
            Ix = identity(self.N[0])
            Iy = identity(self.N[1])
            Iz = identity(self.N[2])
            return kron(kron(Tx,Iy),Iz) + kron(kron(Ix,Ty),Iz) + kron(kron(Ix,Iy), Tz)
        
    def build_potential(self):
        """
        function to build Potential energy (V) operator
        """
        if self.potential is None:
            return diags([np.zeros(np.prod(self.N,))], [0])
        if self.potential_type == "grid":                 
            grids = [np.linspace(-L/2, L/2, N) for L,N in zip(self.extent,self.N)]
            mesh = np.meshgrid(*grids,indexing="ij")
            V_vals = self.potential(*mesh)
            return diags([V_vals.flatten()], [0])
        elif self.potential_type == "matrix":
            return self.potential                              
            
    def solve(self, max_states=5):
        """
         function to solve the lowest energy eigenstates
        """
        energies, states = eigsh(self.H, k=max_states, which = 'SM')
        idx = np.argsort(energies)
        return energies[idx], states[:, idx]
    

