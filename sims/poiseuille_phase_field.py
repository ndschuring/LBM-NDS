import matplotlib.pyplot as plt
from src.lattice import LatticeD2Q9
from src.model import *
from src.utility_functions import mask_from_image, place_circle
from pathlib import Path
import jax.numpy as jnp
import cmasher as cmr
import numpy as np
import jax
import cv2

"""
tester
"""
class Poiseuille(PhaseField):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.rho_bc = kwargs.get('rho_bc')
        self.phi_bc = kwargs.get("phi_bc")
        self.u_bc = kwargs.get("u_bc")

    def __str__(self):
        return "phase_field_poiseuille"

    def apply_bc(self, f, f_prev, **kwargs):
        def zouhe(f_i, f_prev):
            # f_i = f_i.at[0, :, :].set(self.f_equilibrium(self.rho0_ones, self.u_bc, phi=self.phi_bc, p=p)[0, :, :])
            f_i = f_i.at[0, :, :].set(self.f_equilibrium(self.rho0_ones, self.u_bc, phi=self.phi_bc, p=self.rho0_ones+0.001)[0, :, :])
            return f_i
        def neumann_outlet(f_i, f_prev):
            f_i = f_i.at[-1, :, :].set(f_i[-2, :, :])
            f_i = f_i.at[-1, :, :].set(f_i[-2, :, :])
            f_i = f_i.at[-1, :, :].set(f_i[-2, :, :])
            return f_i
        def convective_outlet(f_i, f_prev):
            normal_velocity = u[-2, :, 0][..., jnp.newaxis] #CBC-LV
            # normal_velocity = jnp.max(u[-2, :, 0])[..., jnp.newaxis] #CBC-MV
            # normal_velocity = jnp.average(u[-2, :, 0])[..., jnp.newaxis] #CBC-AV
            f_i = f_i.at[-1, :, :].set((f_prev[-1, :, :]+normal_velocity*f_i[-2, :, :])/(1+normal_velocity))
            return f_i
        def bounce_back_mask(f_i, f_prev):
            f_i = jnp.where(self.bounce_mask, f_prev[..., self.lattice.opp_indices], f_i)
            return f_i
        force = kwargs.get("force")
        g = kwargs.get("g_pop")
        phi = self.get_phi(g)
        rho, u = self.macro_vars(f, force=force, phi=phi)
        # u = jnp.dot(f, self.lattice.c.T) / rho[..., jnp.newaxis] #<-weird behaviour, don't use.
        p = self.get_pressure(rho, f)
        f = zouhe(f, f_prev)
        f = bounce_back_mask(f, f_prev)
        # f = neumann_outlet(f, f_prev)
        f = convective_outlet(f, f_prev)
        return f

    def apply_bc_g(self, g, g_prev, **kwargs):
        def zouhe(g_i, g_prev):
            g_i = g_i.at[0, :, :].set(self.g_equilibrium(self.phi_bc, self.u_bc)[0, :, :])
            return g_i
        def convective_outlet(g_i, g_prev):
            normal_velocity = u[-2, :, 0][..., jnp.newaxis] #CBC-LV
            # normal_velocity = jnp.max(u[-2, :, 0])[..., jnp.newaxis] #CBC-MV
            # normal_velocity = jnp.average(u[-2, :, 0])[..., jnp.newaxis] #CBC-AV
            g_i = g_i.at[-1, :, :].set((g_prev[-1, :, :]+normal_velocity*g_i[-2, :, :])/(1+normal_velocity))
            return g_i
        def neumann_outlet(f_i, f_prev):
            f_i = f_i.at[-1, :, :].set(f_i[-2, :, :])
            f_i = f_i.at[-1, :, :].set(f_i[-2, :, :])
            f_i = f_i.at[-1, :, :].set(f_i[-2, :, :])
            return f_i
        def bounce_back_mask(g_i, g_prev):
            g_i = jnp.where(self.bounce_mask, g_prev[..., self.lattice.opp_indices], g_i)
            return g_i
        f = kwargs.get("f_pop")
        force = kwargs.get("force")
        phi = self.get_phi(g)
        rho, u = self.macro_vars(f, force=force, phi=phi)
        # u = jnp.dot(f, self.lattice.c.T) / rho[..., jnp.newaxis]
        g = bounce_back_mask(g, g_prev)
        g = zouhe(g, g_prev)
        # g = neumann_outlet(g, g_prev)
        g = convective_outlet(g, g_prev)
        return g


if __name__ == "__main__":
    # Define mesh and constants
    nt = int(2e4)
    # nt = int(5e4)
    # nt = 15000
    radius = 15
    rho0 = 1
    tau = 0.6
    lattice = LatticeD2Q9()
    plot_every = 100
    # plot_from = 2500
    plot_from = 0
    # Set collision mask from image
    project_root = Path(__file__).resolve().parent.parent
    image = cv2.imread(str(project_root / "src" / "masks" / "flow10.png"), cv2.IMREAD_GRAYSCALE)
    collision_mask = mask_from_image(image)
    nx, ny = collision_mask.shape
    # nx, ny = 100, 100
    # collision_mask = jnp.zeros((nx, ny), dtype=jnp.bool)
    # collision_mask=collision_mask.at[:, 0].set(True).at[:, -1].set(True)
    # collision_mask= place_circle(collision_mask, radius, value=True, center_x=450)
    u_max = 0.05
    u_bc = jnp.zeros((nx, ny, 2))
    u_bc = u_bc.at[0, :, 0].set(u_max)
    u_innit = jnp.zeros((nx, ny, 2)).at[:,:,0].set(u_max)
    phi_init = jnp.zeros((nx, ny)).at[:, :int(ny / 2)].set(1).at[:, int(ny / 2):].set(0)
    # phi_init = jnp.zeros((nx, ny))
    # phi_init = jnp.ones((nx, ny))
    # phi_init = place_circle(phi_init, radius, value=1, center_x=45)
    # phi_init = phi_init.at[0, :].set(0)
    phi_bc = phi_init
    # collision_mask_debug = np.asarray(collision_mask)
    # Multi-component-specific parameters
    gamma = 1
    param_A = param_B = 4e-5
    kappa = 4.5 * jnp.abs(param_A)
    # kappa = 2.5e-2
    tau_phi = 0.55
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
        'phi_bc': phi_bc,
        'gamma': gamma,
        'param_A': param_A,
        'param_B': param_B,
        'kappa': kappa,
        'tau_phi': tau_phi,
        'phi_init': phi_init,
        'create_video': True,
        'u_bc': u_bc,
    }
    # Create simulation and run
    # simPoiseuille = Poiseuille(**kwargs)
    # simPoiseuille.run(nt)

