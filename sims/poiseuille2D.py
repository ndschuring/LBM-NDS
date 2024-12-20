import matplotlib.pyplot as plt
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
        self.u_max = kwargs.get('u_max')
        self.rho_inlet = jnp.ones((self.nx, self.ny))*kwargs.get('rho_inlet')
        self.rho_outlet = jnp.ones((self.nx, self.ny))*kwargs.get('rho_outlet')
        self.x = jnp.arange(1, self.nx+1) - 0.5
        self.y = jnp.arange(1, self.ny+1) - 0.5
        self.analytical_solution = self.poiseuille_analytical()

    def __str__(self):
        return "Poiseuille_LBMBookBC"

    def apply_bc(self, f, f_prev, force=None):
        def bounce_back_tube2D(f_i, f_prev):
            # Bounce-back top wall
            f_i = f_i.at[:, -1, self.lattice.bottom_indices].set(f_prev[:, -1, self.lattice.top_indices])
            # Bounce-back bottom wall
            f_i = f_i.at[:, 0, self.lattice.top_indices].set(f_prev[:, 0, self.lattice.bottom_indices])
            return f_i
        f = bounce_back_tube2D(f, f_prev)
        return f

    def apply_pre_bc(self, f_post_col, f_pre_col, force=None):
        # Periodic pressure boundary condition
        def inlet_pressure(f_i):
            f_i = f_i.at[0, :, :].set(self.f_equilibrium(self.rho_inlet, u_pre)[-2, :, :] + f_i[-2, :, :] - f_eq[-2, :, :])
            return f_i
        def outlet_pressure(f_i):
            f_i = f_i.at[-1, :, :].set(self.f_equilibrium(self.rho_outlet, u_pre)[2, :, :] + f_i[2, :, :] - f_eq[2, :, :])
            return f_i
        rho_pre, u_pre = self.macro_vars(f_pre_col)
        f_eq = self.f_equilibrium(rho_pre, u_pre)
        f_post_col = inlet_pressure(f_post_col)
        f_post_col = outlet_pressure(f_post_col)
        return f_post_col

    def poiseuille_analytical(self):
        ybottom = 0
        ytop = self.ny
        u_analytical = -4 * self.u_max / (self.ny ** 2) * (self.y - ybottom) * (self.y - ytop)
        return u_analytical

    def plot(self, f, it, **kwargs):
        rho, u = self.macro_vars(f)
        u_magnitude = jnp.linalg.norm(u, axis=-1, ord=2)
        # Plot velocity (magnitude or x-component of velocity vector)
        # plt.imshow(u[:,:,0].T, cmap='viridis')
        plt.imshow(u_magnitude.T, cmap='viridis')
        plt.gca().invert_yaxis()
        plt.colorbar(label="velocity magnitude")
        plt.xlabel("x [lattice units]")
        plt.ylabel("y [lattice units]")
        plt.title("it:" + str(it) + "sum_rho:" + str(jnp.sum(rho)))
        plt.savefig(self.sav_dir + "/fig_2D_it" + str(it) + ".jpg")
        plt.clf()
        # plot 1D velocity profile, along with analytical solution and error
        plt.plot(self.y, u[int(self.nx/2),:,0], label="Poiseuille2D.py")
        plt.plot(self.y, self.analytical_solution, label="analytical solution")
        plt.plot(self.y, self.analytical_solution-u[int(self.nx/2),:,0], label="error")
        plt.ylim(0, self.u_max)
        plt.legend()
        plt.savefig(self.sav_dir + "/fig_1D_it" + str(it) + ".jpg")
        plt.clf()


if __name__ == "__main__":
    time1 = time.time()
    # Define mesh and constants
    nx = 180
    ny = 30
    nt = int(1e4)
    rho0 = 1
    lattice = LatticeD2Q9()
    plot_every = 100
    # set pressure parameters
    nu = (2 * tau - 1) / 6 #kinematic shear velocity
    u_max = 0.1 #maximum velocity
    gradP = 8 * nu * u_max / ny ** 2 #pressure gradient
    rho_outlet = rho0
    rho_inlet = 3 * (nx - 1) * gradP + rho_outlet
    # set kwargs
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
        'plot_every': plot_every,
    }
    # Create simulation and run
    simPoiseuille = Poiseuille(**kwargs)
    simPoiseuille.run(nt)
    time2 = time.time()
    print(time2-time1)