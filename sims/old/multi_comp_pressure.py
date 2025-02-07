import matplotlib.pyplot as plt
from src.lattice import LatticeD2Q9
from src.model import BGKMulti
from src.model import BGK
from src.utility_functions import mask_from_image, place_circle
import jax.numpy as jnp
import numpy as np
import jax
import time
import cv2
import sys

"""
2D poiseuille flow driven by anti-bounce-back inlet/outlet boundary conditions.
Volume contains arbitrary geometry on which no-slip bounce-back is applied.

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
class Poiseuille(BGKMulti):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rho_bc = kwargs.get("rho_bc")
        self.phi_bc = kwargs.get("phi_bc")

    def __str__(self):
        return "multi_comp_pressure"

    def apply_bc(self, f, f_prev, **kwargs):
        def anti_bounce_back_inlet(f_i, f_prev):
            f_i = f_i.at[0, 1:-1, 5].set(-f_prev[0, 1:-1, 7] + 2 * self.lattice.w[7] * self.rho_bc[0, 1:-1] * (
                    1 + ((jnp.tensordot(self.lattice.c[:, 7], u_inlet, axes=(-1, -1)) ** 2) / (
                        2 * self.lattice.cs2 ** 2)) - (jnp.einsum("ij,ij->i", u_inlet, u_inlet)/(2*self.lattice.cs2)) ))

            f_i = f_i.at[0, 1:-1, 1].set(-f_prev[0, 1:-1, 3] + 2 * self.lattice.w[3] * self.rho_bc[0, 1:-1] * (
                    1 + ((jnp.tensordot(self.lattice.c[:, 3], u_inlet, axes=(-1, -1)) ** 2) / (
                    2 * self.lattice.cs2 ** 2)) - (jnp.einsum("ij,ij->i", u_inlet, u_inlet)/(2*self.lattice.cs2)) ))

            f_i = f_i.at[0, 1:-1, 8].set(-f_prev[0, 1:-1, 6] + 2 * self.lattice.w[6] * self.rho_bc[0, 1:-1] * (
                    1 + ((jnp.tensordot(self.lattice.c[:, 6], u_inlet, axes=(-1, -1)) ** 2) / (
                    2 * self.lattice.cs2 ** 2)) - (jnp.einsum("ij,ij->i", u_inlet, u_inlet)/(2*self.lattice.cs2)) ))
            return f_i
        def anti_bounce_back_outlet(f_i, f_prev):
            f_i = f_i.at[-2, 1:-1, 7].set(-f_prev[-2, 1:-1, 5] + 2 * self.lattice.w[5] * self.rho_bc[-2, 1:-1] * (
                    1 + ((jnp.tensordot(self.lattice.c[:, 5], u_outlet, axes=(-1, -1)) ** 2) / (
                        2 * self.lattice.cs2 ** 2)) - (jnp.einsum("ij,ij->i", u_outlet, u_outlet)/(2*self.lattice.cs2)) ))

            f_i = f_i.at[-2, 1:-1, 3].set(-f_prev[-2, 1:-1, 1] + 2 * self.lattice.w[1] * self.rho_bc[-2, 1:-1] * (
                    1 + ((jnp.tensordot(self.lattice.c[:, 1], u_outlet, axes=(-1, -1)) ** 2) / (
                    2 * self.lattice.cs2 ** 2)) - (jnp.einsum("ij,ij->i", u_outlet, u_outlet)/(2*self.lattice.cs2)) ))

            f_i = f_i.at[-2, 1:-1, 6].set(-f_prev[-2, 1:-1, 8] + 2 * self.lattice.w[8] * self.rho_bc[-2, 1:-1] * (
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
        u_inlet  = u[0, 1:-1, :] + 0.5 * (u[ 0, 1:-1, :] - u[ 1, 1:-1, :])
        u_outlet = u[-2, 1:-1, :] + 0.5 * (u[-2, 1:-1, :] - u[-3, 1:-1, :])
        # f = anti_bounce_back_inlet(f, f_prev)
        # f = anti_bounce_back_outlet(f, f_prev)
        # f = neumann_outlet(f, f_prev)
        f = bounce_back_mask(f, f_prev)
        return f

    def apply_bc_g(self, g, g_prev, **kwargs):
        def anti_bounce_back_inlet(g_i, g_prev):
            g_i = g_i.at[0, 1:-1, 5].set(g_prev[0, 1:-1, 7] - 2 * self.lattice.w[7] * self.phi_bc[0, 1:-1] * (
                    1 + ((jnp.tensordot(self.lattice.c[:, 7], u_inlet, axes=(-1, -1)) ** 2) / (
                    2 * self.lattice.cs2 ** 2)) - (jnp.einsum("ij,ij->i", u_inlet, u_inlet) / (2 * self.lattice.cs2))))

            g_i = g_i.at[0, 1:-1, 1].set(g_prev[0, 1:-1, 3] - 2 * self.lattice.w[3] * self.phi_bc[0, 1:-1] * (
                    1 + ((jnp.tensordot(self.lattice.c[:, 3], u_inlet, axes=(-1, -1)) ** 2) / (
                    2 * self.lattice.cs2 ** 2)) - (jnp.einsum("ij,ij->i", u_inlet, u_inlet) / (2 * self.lattice.cs2))))

            g_i = g_i.at[0, 1:-1, 8].set(g_prev[0, 1:-1, 6] - 2 * self.lattice.w[6] * self.phi_bc[0, 1:-1] * (
                    1 + ((jnp.tensordot(self.lattice.c[:, 6], u_inlet, axes=(-1, -1)) ** 2) / (
                    2 * self.lattice.cs2 ** 2)) - (jnp.einsum("ij,ij->i", u_inlet, u_inlet) / (2 * self.lattice.cs2))))
            return g_i

        def anti_bounce_back_outlet(g_i, g_prev):
            g_i = g_i.at[-2, 1:-1, 7].set(-g_prev[-2, 1:-1, 5] + 2 * self.lattice.w[5] * self.phi_bc[-2, 1:-1] * (
                    1 + ((jnp.tensordot(self.lattice.c[:, 5], u_outlet, axes=(-1, -1)) ** 2) / (
                    2 * self.lattice.cs2 ** 2)) - (
                                jnp.einsum("ij,ij->i", u_outlet, u_outlet) / (2 * self.lattice.cs2))))

            g_i = g_i.at[-2, 1:-1, 3].set(-g_prev[-2, 1:-1, 1] + 2 * self.lattice.w[1] * self.phi_bc[-2, 1:-1] * (
                    1 + ((jnp.tensordot(self.lattice.c[:, 1], u_outlet, axes=(-1, -1)) ** 2) / (
                    2 * self.lattice.cs2 ** 2)) - (
                                jnp.einsum("ij,ij->i", u_outlet, u_outlet) / (2 * self.lattice.cs2))))

            g_i = g_i.at[-2, 1:-1, 6].set(-g_prev[-2, 1:-1, 8] + 2 * self.lattice.w[8] * self.phi_bc[-2, 1:-1] * (
                    1 + ((jnp.tensordot(self.lattice.c[:, 8], u_outlet, axes=(-1, -1)) ** 2) / (
                    2 * self.lattice.cs2 ** 2)) - (
                                jnp.einsum("ij,ij->i", u_outlet, u_outlet) / (2 * self.lattice.cs2))))
            return g_i

        def neumann_outlet(g_i, g_prev):
            g_i = g_i.at[-1, :, 6].set(g_prev[-2, :, 6])
            g_i = g_i.at[-1, :, 3].set(g_prev[-2, :, 3])
            g_i = g_i.at[-1, :, 7].set(g_prev[-2, :, 7])
            return g_i

        def bounce_back_mask(g_i, g_prev):
            g_i = jnp.where(self.bounce_mask, g_prev[..., self.lattice.opp_indices], g_i)
            return g_i

        f = kwargs.get("f_pop")
        rho, u = self.macro_vars(f)
        phi, _ = self.macro_vars(g)
        # if self.debug:
        #     debug_f = np.asarray(f)
        #     debug_rho = np.asarray(rho)
        #     debug_u = np.asarray(u)
        #     debug_phi = np.asarray(phi)
        u_inlet = u[0, 1:-1, :] + 0.5 * (u[0, 1:-1, :] - u[1, 1:-1, :])
        u_outlet = u[-2, 1:-1, :] + 0.5 * (u[-2, 1:-1, :] - u[-3, 1:-1, :])
        # g = anti_bounce_back_inlet(g, g_prev)
        # g = anti_bounce_back_outlet(g, g_prev)
        # g = neumann_outlet(g, g_prev)
        g = bounce_back_mask(g, g_prev)
        return g


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
    # nt = int(1e3)
    # nt = int(1e2)
    rho0 = 1.0
    tau = 1.0
    lattice = LatticeD2Q9()
    plot_every = 1
    plot_from = 0
    # Set collision mask from image
    image = cv2.imread('/src/masks/flow2.png', cv2.IMREAD_GRAYSCALE)
    collision_mask = mask_from_image(image)
    nx, ny = collision_mask.shape
    # initialise rho_bc, a matrix mask specifying which densities need to be enforced at certain coordinates
    rho_bc = jnp.ones((nx, ny))*rho0
    rho_bc = rho_bc.at[0, :].set(1.1)
    ## set top and bottom to two phases
    phi_init = jnp.zeros((nx, ny)).at[:, :int(ny / 2)].set(1)
    phi_init = phi_init.at[:, int(ny / 2):].set(-1)
    ## set left and right to two phases
    # phi_init = jnp.zeros((nx, ny)).at[:int(nx/2), :].set(1)
    # phi_init = phi_init.at[int(nx / 2):, :].set(-1)
    ## set entire domain to 0
    # phi_init = jnp.zeros((nx, ny))
    ## set entire domain to -1 or 1 or in between.
    phi_init = jnp.ones((nx, ny)).at[:,:].set(-1)
    phi_init = place_circle(phi_init, 15, 1)
    ## set enforcement rules equal to initialisation.
    phi_bc = phi_init

    # Multi-component-specific parameters
    gamma = 1
    param_A = param_B = -4e-4
    # kappa = 4.5 * jnp.abs(param_A)
    kappa = 2.5e-2
    tau_phi = 1
    # Set kwargs
    kwargs = {
        'lattice': lattice,
        'tau': tau,
        'rho0': rho0,
        'plot_every': plot_every,
        'plot_from': plot_from,
        'collision_mask': collision_mask,
        'rho_bc': rho_bc,
        'phi_bc': phi_bc,
        'gamma': gamma,
        'param_A': param_A,
        'param_B': param_B,
        'kappa': kappa,
        'tau_phi': tau_phi,
        'phi_init': phi_init,
    }
    # Create simulation and run
    simPoiseuille = Poiseuille(**kwargs)
    simPoiseuille.run(nt)
    time2 = time.time()
    print(time2-time1)

