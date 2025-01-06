import matplotlib.pyplot as plt
from src.lattice import LatticeD2Q9
import jax.numpy as jnp
from src.model import BGKMulti
import time
"""
Couette Flow
                Generalised
                moving Wall BC (Krüger et al.)
        +-------------------------------------------+
        |                  ------>                  |
Periodic|                                           |Periodic
    BC  |                                           |   BC
        |                                           |
        |                                           |
      (0,0)-----------------------------------------+
                        No slip BC
"""

class Couette(BGKMulti):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.u_bc = kwargs.get('u_bc')

    def __str__(self):
        return "Multi-component_test"


def initialize_grid(nx, ny, r):
    """
    Initialize a grid of dimensions (nx, ny) with -1 everywhere, and set points
    within a circle of radius r at the center to 1.
    :param nx: Number of rows in the grid.
    :param ny: Number of columns in the grid.
    :param r: Radius of the circle at the center.
    :return: jnp.ndarray: A 2D array representing the initialized grid.
    """
    # Create a grid of indices
    x = jnp.arange(nx)
    y = jnp.arange(ny)
    xx, yy = jnp.meshgrid(x, y, indexing='ij')
    # Compute the distance from the center
    center_x, center_y = nx // 2, ny // 2
    distance_from_center = jnp.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    # Initialize the grid with -1
    grid = jnp.full((nx, ny), -1, dtype=jnp.float32)
    # Set points within the circle to 1
    grid = jnp.where(distance_from_center <= r, 1, grid)
    return grid

if __name__ == "__main__":
    time1 = time.time()
    # Define mesh and constants
    nx = 180
    ny = 40
    nt = int(3e4)
    r = 12
    rho0 = 1
    tau = 1
    lattice = LatticeD2Q9()
    plot_every = 1000
    plot_from = 0
    # Initialise u_bc, a matrix mask specifying which velocities need to be enforced at certain coordinates
    u_bc = jnp.zeros((nx, ny, 2))
    u_top_wall = 0.1
    u_bc = u_bc.at[:, -1, 0].set(u_top_wall)
    ### Initialisation of phi, various methods
    # -> initialise as 2 halves of fluid:
    phi_init = jnp.zeros((nx, ny)).at[:, :int(ny/2)].set(1)
    phi_init = phi_init.at[:, int(ny/2):].set(-1)
    # -> initialise as zero-matrix
    # phi_init = jnp.zeros_like(phi_init)
    # -> initialise as circle
    # phi_init = initialize_grid(nx, ny, r)
    # -> initialise as 1 corner to other fluid (why did I make this?)
    # phi_init = phi_init.at[:,:].set(1)
    # phi_init = phi_init.at[int(nx/2), int(ny/2)].set(-1)
    # Multi-component-specific parameters
    gamma = 1
    param_A = param_B = -4e-5
    kappa = 4.5*jnp.abs(param_A)
    # kappa = 2.5e-3
    tau_phi = 1
    # Set kwargs
    kwargs = {
        'lattice': lattice,
        'tau': tau,
        'nx': nx,
        'ny': ny,
        'rho0': rho0,
        'plot_every': plot_every,
        'plot_from': plot_from,
        'u_bc': u_bc,
        'gamma': gamma,
        'param_A': param_A,
        'param_B': param_B,
        'kappa': kappa,
        'tau_phi': tau_phi,
        'phi_init': phi_init,
    }
    # Create simulation and run
    sim = Couette(**kwargs)
    sim.run(nt)
    time2 = time.time()
    print(time2-time1)