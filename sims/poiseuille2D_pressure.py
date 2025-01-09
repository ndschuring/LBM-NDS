import matplotlib.pyplot as plt
from src.lattice import LatticeD2Q9
from src.model import BGK
import jax.numpy as jnp
import jax
import time

"""
2D poiseuille flow driven by moving wall inlet/outlet boundary conditions
"""
class Poiseuille(BGK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return "Poiseuille_pressure"

    def apply_bc(self, f, f_prev, **kwargs):
        def bounce_back_tube(f_i, f_prev):
            # Bounce-back top wall
            f_i = f_i.at[:, -1, self.lattice.bottom_indices].set(
                f_prev[:, -1, self.lattice.opp_indices[self.lattice.bottom_indices]])
            # Bounce-back bottom wall
            f_i = f_i.at[:, 0, self.lattice.top_indices].set(
                f_prev[:, 0, self.lattice.opp_indices[self.lattice.top_indices]])
            return f_i
        def anti_bounce_back_inlet(f_i, f_prev):
            f_i = f_i.at[0, :, 5].set(-f_prev[0, :, 7] + 2 * self.lattice.w[7] * rho_bc[0, :] * (
                    1 + ((jnp.tensordot(self.lattice.c[:, 7], u_inlet, axes=(-1, -1)) ** 2) / (
                        2 * self.lattice.cs2 ** 2)) - (jnp.einsum("ij,ij->i", u_inlet, u_inlet)/(2*self.lattice.cs2)) ))

            f_i = f_i.at[0, :, 1].set(-f_prev[0, :, 3] + 2 * self.lattice.w[3] * rho_bc[0, :] * (
                    1 + ((jnp.tensordot(self.lattice.c[:, 3], u_inlet, axes=(-1, -1)) ** 2) / (
                    2 * self.lattice.cs2 ** 2)) - (jnp.einsum("ij,ij->i", u_inlet, u_inlet)/(2*self.lattice.cs2)) ))

            f_i = f_i.at[0, :, 8].set(-f_prev[0, :, 6] + 2 * self.lattice.w[6] * rho_bc[0, :] * (
                    1 + ((jnp.tensordot(self.lattice.c[:, 6], u_inlet, axes=(-1, -1)) ** 2) / (
                    2 * self.lattice.cs2 ** 2)) - (jnp.einsum("ij,ij->i", u_inlet, u_inlet)/(2*self.lattice.cs2)) ))
            return f_i
        def anti_bounce_back_outlet(f_i, f_prev):
            f_i = f_i.at[-2, :, 7].set(-f_prev[-2, :, 5] + 2 * self.lattice.w[5] * rho_bc[-2, :] * (
                    1 + ((jnp.tensordot(self.lattice.c[:, 5], u_outlet, axes=(-1, -1)) ** 2) / (
                        2 * self.lattice.cs2 ** 2)) - (jnp.einsum("ij,ij->i", u_outlet, u_outlet)/(2*self.lattice.cs2)) ))

            f_i = f_i.at[-2, :, 3].set(-f_prev[-2, :, 1] + 2 * self.lattice.w[1] * rho_bc[-2, :] * (
                    1 + ((jnp.tensordot(self.lattice.c[:, 1], u_outlet, axes=(-1, -1)) ** 2) / (
                    2 * self.lattice.cs2 ** 2)) - (jnp.einsum("ij,ij->i", u_outlet, u_outlet)/(2*self.lattice.cs2)) ))

            f_i = f_i.at[-2, :, 6].set(-f_prev[-2, :, 8] + 2 * self.lattice.w[8] * rho_bc[-2, :] * (
                    1 + ((jnp.tensordot(self.lattice.c[:, 8], u_outlet, axes=(-1, -1)) ** 2) / (
                    2 * self.lattice.cs2 ** 2)) - (jnp.einsum("ij,ij->i", u_outlet, u_outlet)/(2*self.lattice.cs2)) ))
            return f_i
        def neumann_outlet(f_i, f_prev):
            f_i = f_i.at[-1, :, 6].set(f_prev[-2, :, 6])
            f_i = f_i.at[-1, :, 3].set(f_prev[-2, :, 3])
            f_i = f_i.at[-1, :, 7].set(f_prev[-2, :, 7])
            return f_i

        rho, u = self.macro_vars(f)
        u_inlet  = u[0, :, :] + 0.5 * (u[ 0, :, :] - u[ 1, :, :])
        u_outlet = u[-2, :, :] + 0.5 * (u[-2, :, :] - u[-3, :, :])
        f = anti_bounce_back_inlet(f, f_prev)
        f = anti_bounce_back_outlet(f, f_prev)
        f = bounce_back_tube(f, f_prev)
        f = neumann_outlet(f, f_prev)
        return f

# def poiseuille_analytical():
#     y = jnp.arange(1, ny + 1) - 0.5
#     ybottom = 0
#     ytop = ny
#     u_analytical = -4 * u_in / (ny ** 2) * (y - ybottom) * (y - ytop)
#     return u_analytical


def initialize_grid(nx, ny, r):
    """
    Initialize a boolean grid of dimensions (nx, ny) with False everywhere, and set points
    within a circle of radius r at the center to True.
    :param nx: Number of rows in the grid.
    :param ny: Number of columns in the grid.
    :param r: Radius of the circle at the center.
    :return: jnp.ndarray: A 2D boolean array representing the initialized grid.
    """
    # Create a grid of indices
    x = jnp.arange(nx)
    y = jnp.arange(ny)
    xx, yy = jnp.meshgrid(x, y, indexing='ij')
    # Compute the distance from the center
    center_x, center_y = nx // 2, ny // 2
    distance_from_center = jnp.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    # Create a boolean grid where True is inside the circle and False is outside
    grid = distance_from_center <= r
    return grid

if __name__ == "__main__":
    time1 = time.time()
    # Define mesh and constants
    nx = 250
    ny = 50
    nt = int(1e4)
    rho0 = 1
    tau = 1
    lattice = LatticeD2Q9()
    plot_every = 100
    plot_from = 0
    # initialise rho_bc, a matrix mask specifying which densities need to be enforced at certain coordinates
    rho_bc = jnp.ones((nx, ny))*rho0
    rho_bc = rho_bc.at[0, :].set(1.05)
    # Set kwargs
    kwargs = {
        'lattice': lattice,
        'tau': tau,
        'nx': nx,
        'ny': ny,
        'rho0': rho0,
        'plot_every': plot_every,
        'plot_from': plot_from,
        'debug': (True, 299),
    }
    # Create simulation and run
    simPoiseuille = Poiseuille(**kwargs)
    simPoiseuille.run(nt)
    time2 = time.time()
    print(time2-time1)

