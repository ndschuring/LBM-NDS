from src.boundary_conditions import *
from src.lattice import LatticeD2Q9
from src.model import BGK
import jax.numpy as jnp
import time
"""
Poiseuille Flow

                        No slip BC
        +-------------------------------------------+
        |                                           |
Periodic|                                           |Periodic
Pressure|       ------------>                       |pressure
   BC   |                                           |   BC
        |                                           |
        +-------------------------------------------+
                        No slip BC
"""



class Poiseuille(BGK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.u_max = kwargs.get('u_max')
        # self.gradP = kwargs.get('gradP')
        self.rho_inlet = jnp.ones((self.nx, self.ny))*kwargs.get('rho_inlet')
        self.rho_outlet = jnp.ones((self.nx, self.ny))*kwargs.get('rho_outlet')

    def apply_bc(self, f):
        def bounce_back_tube2D(f_i):
            # Bounce-back top wall
            f_i = f_i.at[:, -1, self.lattice.bottom_indices].set(f_i[:, -1, self.lattice.top_indices])
            # Bounce-back bottom wall
            f_i = f_i.at[:, -1, self.lattice.top_indices].set(f_i[:, -1, self.lattice.bottom_indices])
            return f_i
        f = bounce_back_tube2D(f)
        return f

    def apply_pre_bc(self, f_post_col, f_pre_col):
        def inlet_pressure(f_i):
            f_i = f_i.at[0, :, :].set(self.equilibrium(self.rho_inlet, u_pre)[-2,:,:]+f_i[-2, :, :]-f_eq[-2,:,:])
            return f_i
        def outlet_pressure(f_i):
            f_i = f_i.at[-1, :, :].set(self.equilibrium(self.rho_outlet, u_pre)[2, :, :] + f_i[2, :, :]-f_eq[2, :, :])
            return f_i
        rho_pre, u_pre = self.macro_vars(f_pre_col)
        # rho, u = self.macro_vars(f_post_col)
        f_eq = self.equilibrium(rho_pre, u_pre)
        f_post_col = inlet_pressure(f_post_col)
        f_post_col = outlet_pressure(f_post_col)
        return f_post_col


if __name__ == "__main__":
    time1 = time.time()
    nx = 180
    ny = 30
    nt = int(1e4)
    rho0 = 1
    # tau = 1
    lattice = LatticeD2Q9()

    tau = (3/16)**0.5 + 0.5             #relaxation time
    nu = (2*tau-1)/6                    #kinematic shear velocity
    u_max = 0.1                         #maximum velocity
    gradP = 8 * nu * u_max / ny ** 2    #pressure gradient
    rho_outlet = rho0
    rho_inlet = 3 * (nx - 1) * gradP + rho_outlet

    kwargs = {
        'lattice': lattice,
        'tau': tau,
        'nx': nx,
        'ny': ny,
        'rho0': rho0,
        'u_max': u_max,
        'gradP': gradP,
        'rho_outlet': rho_outlet,
        'rho_inlet': rho_inlet,
    }
    simPoiseuille = Poiseuille(**kwargs)
    simPoiseuille.run(nt)
    time2 = time.time()
    print(time2-time1)