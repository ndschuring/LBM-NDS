import matplotlib.pyplot as plt
from src.lattice import LatticeD2Q9
from src.model import BGK
from src.utility_functions import mask_from_image
import jax.numpy as jnp
import numpy as np
import jax
import time
import cv2

"""


                                        No-Slip BC
                       +-------------------------------------------+
                       |                                           |
Pressure dirichlet     |                                           |        Pressure dirichlet BC (anti-bounce-back)
      BC        -----> |                                           | -----> Neumann zero-gradient BC
(anti-bounce-back)     |                                           | 
                       |                                           |
                     (0,0)-----------------------------------------+
                                        No-Slip BC
"""
class Poiseuille(BGK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rho_bc = kwargs.get('rho_bc')

    def __str__(self):
        return "Herringbone2D"

    def apply_bc(self, f, f_prev, **kwargs):
        def anti_bounce_back_inlet(f_i, f_prev):
            f_i = f_i.at[0, 25:50, 5].set(-f_prev[0, 25:50, 7] + 2 * self.lattice.w[7] * self.rho_bc[0, 25:50] * (
                    1 + ((jnp.tensordot(self.lattice.c[:, 7], u_inlet, axes=(-1, -1)) ** 2) / (
                        2 * self.lattice.cs2 ** 2)) - (jnp.einsum("ij,ij->i", u_inlet, u_inlet)/(2*self.lattice.cs2)) ))

            f_i = f_i.at[0, 25:50, 1].set(-f_prev[0, 25:50, 3] + 2 * self.lattice.w[3] * self.rho_bc[0, 25:50] * (
                    1 + ((jnp.tensordot(self.lattice.c[:, 3], u_inlet, axes=(-1, -1)) ** 2) / (
                    2 * self.lattice.cs2 ** 2)) - (jnp.einsum("ij,ij->i", u_inlet, u_inlet)/(2*self.lattice.cs2)) ))

            f_i = f_i.at[0, 25:50, 8].set(-f_prev[0, 25:50, 6] + 2 * self.lattice.w[6] * self.rho_bc[0, 25:50] * (
                    1 + ((jnp.tensordot(self.lattice.c[:, 6], u_inlet, axes=(-1, -1)) ** 2) / (
                    2 * self.lattice.cs2 ** 2)) - (jnp.einsum("ij,ij->i", u_inlet, u_inlet)/(2*self.lattice.cs2)) ))
            return f_i
        def anti_bounce_back_outlet(f_i, f_prev):
            f_i = f_i.at[-2, :, 7].set(-f_prev[-2, :, 5] + 2 * self.lattice.w[5] * self.rho_bc[-2, :] * (
                    1 + ((jnp.tensordot(self.lattice.c[:, 5], u_outlet, axes=(-1, -1)) ** 2) / (
                        2 * self.lattice.cs2 ** 2)) - (jnp.einsum("ij,ij->i", u_outlet, u_outlet)/(2*self.lattice.cs2)) ))

            f_i = f_i.at[-2, :, 3].set(-f_prev[-2, :, 1] + 2 * self.lattice.w[1] * self.rho_bc[-2, :] * (
                    1 + ((jnp.tensordot(self.lattice.c[:, 1], u_outlet, axes=(-1, -1)) ** 2) / (
                    2 * self.lattice.cs2 ** 2)) - (jnp.einsum("ij,ij->i", u_outlet, u_outlet)/(2*self.lattice.cs2)) ))

            f_i = f_i.at[-2, :, 6].set(-f_prev[-2, :, 8] + 2 * self.lattice.w[8] * self.rho_bc[-2, :] * (
                    1 + ((jnp.tensordot(self.lattice.c[:, 8], u_outlet, axes=(-1, -1)) ** 2) / (
                    2 * self.lattice.cs2 ** 2)) - (jnp.einsum("ij,ij->i", u_outlet, u_outlet)/(2*self.lattice.cs2)) ))
            return f_i
        def neumann_outlet(f_i, f_prev):
            f_i = f_i.at[-1, :, 6].set(f_prev[-2, :, 6])
            f_i = f_i.at[-1, :, 3].set(f_prev[-2, :, 3])
            f_i = f_i.at[-1, :, 7].set(f_prev[-2, :, 7])
            return f_i
        def bounce_back_mask(f_i, f_prev):
            f_i = jnp.where(self.bounce_mask, f_prev[..., self.lattice.opp_indices], f_i)
            return f_i

        rho, u = self.macro_vars(f)
        u_inlet  = u[0, 25:50, :] + 0.5 * (u[ 0, 25:50, :] - u[ 1, 25:50, :])
        u_outlet = u[-2, :, :] + 0.5 * (u[-2, :, :] - u[-3, :, :])
        f = bounce_back_mask(f, f_prev)
        f = anti_bounce_back_inlet(f, f_prev)
        f = anti_bounce_back_outlet(f, f_prev)
        f = neumann_outlet(f, f_prev)
        return f

# def poiseuille_analytical():
#     y = jnp.arange(1, ny + 1) - 0.5
#     ybottom = 0
#     ytop = ny
#     u_analytical = -4 * u_in / (ny ** 2) * (y - ybottom) * (y - ytop)
#     return u_analytical

if __name__ == "__main__":
    time1 = time.time()
    # Define mesh and constants
    nt = int(1e4)
    rho0 = 1
    tau = 1.2
    lattice = LatticeD2Q9()
    plot_every = 100
    plot_from = 0
    # Set collision mask from image
    image = cv2.imread('C:/Users/ndsch/PycharmProjects/LBM-NDS/src/masks/flow6.png', cv2.IMREAD_GRAYSCALE)
    collision_mask = mask_from_image(image)
    nx, ny = collision_mask.shape
    # initialise rho_bc, a matrix mask specifying which densities need to be enforced at certain coordinates
    rho_bc = jnp.ones((nx, ny))*rho0
    rho_bc = rho_bc.at[0, :].set(1.1)
    # Set kwargs
    kwargs = {
        'lattice': lattice,
        'tau': tau,
        'rho0': rho0,
        'plot_every': plot_every,
        'plot_from': plot_from,
        'collision_mask': collision_mask,
        'rho_bc': rho_bc,
    }
    # Create simulation and run
    simPoiseuille = Poiseuille(**kwargs)
    simPoiseuille.run(nt)
    time2 = time.time()
    print(time2-time1)

