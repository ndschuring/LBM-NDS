import matplotlib.pyplot as plt
from src.lattice import LatticeD2Q9
from src.model import BGK
from src.utility_functions import mask_from_image, place_circle
import jax.numpy as jnp
import cmasher as cmr
import numpy as np
import jax
import cv2

"""
Van Karman Vortex Street
2D flow driven by velocity Zou-He inlet BC and convective outlet BC.
Fluid flows around circular object, creating vortices after approximately 6000 iterations
Initial velocity applied to entire domain

                                        Periodic
                       +-------------------------------------------+
                       |                                           |
Zou-he velocity        |                    000                    |        
  inlet BC      -----> |                   00000                   | -----> Convective outlet BC
                       |                    000                    |            (CBC-LV)
                       |                                           |
                     (0,0)-----------------------------------------+
                                        Periodic
"""
class Poiseuille(BGK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rho_bc = kwargs.get('rho_bc')

    def __str__(self):
        return "Vortex_Street"

    def apply_bc(self, f, f_prev, **kwargs):
        # def nebb_inlet(f_i):
        #     Ny = (1/2)*(f_i[0, 1:-1, 4]-f_i[0, 1:-1, 2]) + (1/3)*rho_bc[0, 1:-1]*u_bc[0, 1:-1, 1]
        #     f_i = f_i.at[0, 1:-1, 1].set(f_i[0, 1:-1, 3]+(2/3)*rho_bc[0, 1:-1]*u_bc[0, 1:-1, 0])
        #     f_i = f_i.at[0, 1:-1, 5].set(f_i[0, 1:-1, 7]+(1/6)*rho_bc[0, 1:-1]*(u_bc[0, 1:-1, 0]+u_bc[0, 1:-1, 1])+Ny)
        #     f_i = f_i.at[0, 1:-1, 8].set(f_i[0, 1:-1, 6]-(1/6)*rho_bc[0, 1:-1]*(-u_bc[0, 1:-1, 0]+u_bc[0, 1:-1, 1])-Ny)
        #     return f_i
        # def zouhe(f_i, f_prev):
        #     f_i = f_i.at[0, 1:-1, self.lattice.right_indices].set(self.f_equilibrium(self.rho0_ones, u_bc)[0, 1:-1, self.lattice.right_indices])
        #     return f_i
        def zouhe(f_i, f_prev):
            f_i = f_i.at[0, 1:-1, :].set(self.f_equilibrium(self.rho0_ones, u_bc)[0, 1:-1, :])
            return f_i
        # def neumann_outlet(f_i, f_prev):
        #     f_i = f_i.at[-1, :, 6].set(f_prev[-2, :, 6])
        #     f_i = f_i.at[-1, :, 3].set(f_prev[-2, :, 3])
        #     f_i = f_i.at[-1, :, 7].set(f_prev[-2, :, 7])
        #     return f_i
        def neumann_outlet(f_i, f_prev):
            f_i = f_i.at[-1, :, :].set(f_prev[-2, :, :])
            f_i = f_i.at[-1, :, :].set(f_prev[-2, :, :])
            f_i = f_i.at[-1, :, :].set(f_prev[-2, :, :])
            return f_i
        def convective_outlet(f_i, f_prev):
            normal_velocity = u[-2, :, 0][..., jnp.newaxis] #CBC-LV
            # normal_velocity = jnp.max(u[-2, :, 0])[..., jnp.newaxis] #CBC-MV
            # normal_velocity = jnp.average(u[-2, :, 0])[..., jnp.newaxis] #CBC-AV
            f_i = f_i.at[-1, :, :].set((f_prev[-1, :, :]+normal_velocity*f_i[-2, :, :])/(1+normal_velocity))
            return f_i
        # def neumann_outlet(f_i, f_prev):
        #     f_i = f_i.at[-1, 1:-1, :].set(f_prev[-2, 1:-1, :])
        #     f_i = f_i.at[-1, 1:-1, :].set(f_prev[-2, 1:-1, :])
        #     f_i = f_i.at[-1, 1:-1, :].set(f_prev[-2, 1:-1, :])
        #     return f_i
        def bounce_back_mask(f_i, f_prev):
            f_i = jnp.where(self.bounce_mask, f_prev[..., self.lattice.opp_indices], f_i)
            return f_i

        rho, u = self.macro_vars(f)
        f = bounce_back_mask(f, f_prev)
        f = zouhe(f, f_prev)
        # f = neumann_outlet(f, f_prev)
        f = convective_outlet(f, f_prev)
        return f


# def poiseuille_analytical():
#     y = jnp.arange(1, ny + 1) - 0.5
#     ybottom = 0
#     ytop = ny
#     u_analytical = -4 * u_in / (ny ** 2) * (y - ybottom) * (y - ytop)
#     return u_analytical

if __name__ == "__main__":
    # Define mesh and constants
    nt = int(1e4)
    nt = 15000
    radius = 5
    rho0 = 1
    tau = 0.5075
    lattice = LatticeD2Q9()
    plot_every = 100
    plot_from = 0
    # Set collision mask from image
    # image = cv2.imread('C:/Users/ndsch/PycharmProjects/LBM-NDS/src/masks/flow10.png', cv2.IMREAD_GRAYSCALE)
    # collision_mask = mask_from_image(image)
    # collision_mask = place_circle(collision_mask, 4)
    # nx, ny = collision_mask.shape
    nx, ny = 300, 50
    collision_mask = jnp.zeros((nx, ny), dtype=jnp.bool)
    collision_mask = place_circle(collision_mask, 5, center_x=50)
    u_max = 0.04
    u_bc = jnp.zeros((nx, ny, 2))
    u_bc = u_bc.at[0, :, 0].set(u_max)
    u_innit = jnp.zeros((nx, ny, 2)).at[:,:,0].set(u_max)
    # collision_mask_debug = np.asarray(collision_mask)
    # Set kwargs
    kwargs = {
        'lattice': lattice,
        'tau': tau,
        'rho0': rho0,
        'plot_every': plot_every,
        'plot_from': plot_from,
        'collision_mask': collision_mask,
        'length_scale': radius,
        'u_max': u_max,
        'u_innit': u_innit,
    }
    # Create simulation and run
    simPoiseuille = Poiseuille(**kwargs)
    simPoiseuille.run(nt)

