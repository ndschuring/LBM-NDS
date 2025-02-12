import matplotlib.pyplot as plt
from src.lattice import LatticeD2Q9
import jax.numpy as jnp
from src.model import BGK
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

class Couette(BGK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.u_bc = kwargs.get('u_bc')

    def __str__(self):
        return "Couette_Flow_alt"

    def apply_bc(self, f, f_prev, **kwargs):
        def bounce_back_couette2D(f_i, f_prev):
            # Bounce-back top wall
            f_i = f_i.at[:, -1, self.lattice.bottom_indices].set(
                f_prev[:, -1, self.lattice.opp_indices[self.lattice.bottom_indices]])
            # Bounce-back bottom wall
            f_i = f_i.at[:, 0, self.lattice.top_indices].set(
                f_prev[:, 0, self.lattice.opp_indices[self.lattice.top_indices]])
            return f_i
        def moving_wall_correction(f_i):
            # top wall
            f_i = f_i.at[:, -1, 7].add(- 2 * self.lattice.w[5] * rho[:, -1] * (
                        jnp.tensordot(self.lattice.c[:, 5], u_bc[:, -1], axes=(-1, -1)) / self.lattice.cs2))
            f_i = f_i.at[:, -1, 4].add(- 2 * self.lattice.w[2] * rho[:, -1] * (
                        jnp.tensordot(self.lattice.c[:, 2], u_bc[:, -1], axes=(-1, -1)) / self.lattice.cs2))
            f_i = f_i.at[:, -1, 8].add(- 2 * self.lattice.w[6] * rho[:, -1] * (
                        jnp.tensordot(self.lattice.c[:, 6], u_bc[:, -1], axes=(-1, -1)) / self.lattice.cs2))
            return f_i
        rho, u = self.macro_vars(f)
        f_i = bounce_back_couette2D(f, f_prev)
        f_i = moving_wall_correction(f_i)
        return f_i

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
        plt.plot(u[int(self.nx / 2),:, 0].T, self.y,  label="Velocity Profile nx=nx/2")
        plt.plot(couette_analytical().T, self.y, label="Analytical")
        plt.plot(jnp.abs(couette_analytical()-u[int(self.nx / 2),:, 0]).T, self.y, label="error")
        plt.xlabel("velocity magnitude")
        plt.ylabel("y [lattice units]")
        plt.legend()
        plt.savefig(self.sav_dir + "/fig_1D_it" + str(it) + ".jpg")
        plt.clf()

def couette_analytical():
    y = jnp.arange(1, ny + 1) - 0.5
    u_analytical = u_top_wall / ny*y
    return u_analytical

if __name__ == "__main__":
    time1 = time.time()
    # Define mesh and constants
    nx = 180
    ny = 30
    nt = int(1e4)
    rho0 = 1
    tau = 1
    lattice = LatticeD2Q9()
    plot_every = 100
    # Initialise u_bc, a matrix mask specifying which velocities need to be enforced at certain coordinates
    u_bc = jnp.zeros((nx, ny, 2))
    u_top_wall = 0.1
    u_bc = u_bc.at[:, -1, 0].set(u_top_wall)
    # Set kwargs
    kwargs = {
        'lattice': lattice,
        'tau': tau,
        'nx': nx,
        'ny': ny,
        'rho0': rho0,
        'plot_every': plot_every,
        'u_bc': u_bc,
    }
    # Create simulation and run
    sim = Couette(**kwargs)
    sim.run(nt)
    time2 = time.time()
    print(time2-time1)