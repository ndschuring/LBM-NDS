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

    def plot(self, f, it, **kwargs):
        g = kwargs.get('g')
        rho, u = self.macro_vars(f)
        phi, _ = self.macro_vars(g)
        print(u[int(self.nx / 2), :, 0].mean())
        u_magnitude = jnp.linalg.norm(u, axis=-1, ord=2)
        # plt.imshow(u[:,:,0].T, cmap='viridis')
        plt.imshow(u_magnitude.T, cmap='viridis')
        # u_masked = jnp.where(wall_mask, 0, u_magnitude)
        # plt.imshow(u_masked.T, cmap='viridis')
        plt.gca().invert_yaxis()
        plt.colorbar(label="velocity magnitude")
        plt.xlabel("x [lattice units]")
        plt.ylabel("y [lattice units]")
        plt.title("it:" + str(it) + " sum_rho:" + str(jnp.sum(rho[1:-1, 1:-1])))
        plt.savefig(self.sav_dir + "/fig_2D_it" + str(it) + ".jpg", dpi=100)
        plt.clf()
        plt.imshow(phi.T, cmap='viridis', vmin=-1, vmax=1)
        # phi_masked = jnp.where(wall_mask, 0, phi)
        # plt.imshow(phi_masked.T, cmap='viridis')
        plt.gca().invert_yaxis()
        plt.colorbar(label="Order Parameter")
        plt.xlabel("x [lattice units]")
        plt.ylabel("y [lattice units]")

        plt.title("it:"+str(it)+" Order parameter phi")
        plt.savefig(self.sav_dir+"/fig_2D_phi_it"+str(it)+".jpg", dpi=100)
        plt.clf()

def couette_analytical():
    y = jnp.arange(1, ny + 1) - 0.5
    ybottom = 0
    ytop = ny
    u_analytical = u_top_wall / ny*y
    return u_analytical


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
    phi_init = initialize_grid(nx, ny, r)
    rho0 = 1
    tau = 1
    lattice = LatticeD2Q9()
    plot_every = 100
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
    param_A = param_B = 4e-5
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