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
        +-------------------------------------------+
                        No slip BC
"""

class Couette(BGK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.u_bc = kwargs.get('u_bc')

    def __str__(self):
        return "Couette_Flow_multi"

    def apply_bc(self, f, f_prev, force=None):
        def bounce_back_couette2D(f_i, f_prev):
            # Bounce-back top wall
            f_i = f_i.at[:, -1, 7].set(f_prev[:, -1, 5])# - 2 * self.lattice.w[5]*rho[:,-1]*(jnp.tensordot(self.lattice.c[:,5], u_bc[:, -1], axes=(-1, -1))/self.lattice.cs2))
            f_i = f_i.at[:, -1, 4].set(f_prev[:, -1, 2])# - 2 * self.lattice.w[2]*rho[:,-1]*(jnp.tensordot(self.lattice.c[:,2], u_bc[:, -1], axes=(-1, -1))/self.lattice.cs2))
            f_i = f_i.at[:, -1, 8].set(f_prev[:, -1, 6])# - 2 * self.lattice.w[6]*rho[:,-1]*(jnp.tensordot(self.lattice.c[:, 6], u_bc[:, -1], axes=(-1, -1))/self.lattice.cs2))
            # Bounce-back bottom wall
            f_i = f_i.at[:,  0, 6].set(f_prev[:,  0, 8])
            f_i = f_i.at[:,  0, 2].set(f_prev[:,  0, 4])
            f_i = f_i.at[:,  0, 5].set(f_prev[:,  0, 7])
            return f_i
        def moving_wall_correction(f_i):
            # top wall
            f_i = f_i.at[:, -1, 7].add( - 2 * self.lattice.w[5]*self.rho0_ones[:,-1]*(jnp.tensordot(self.lattice.c[:,5], u_bc[:, -1], axes=(-1, -1))/self.lattice.cs2))
            f_i = f_i.at[:, -1, 4].add( - 2 * self.lattice.w[2]*self.rho0_ones[:,-1]*(jnp.tensordot(self.lattice.c[:,2], u_bc[:, -1], axes=(-1, -1))/self.lattice.cs2))
            f_i = f_i.at[:, -1, 8].add( - 2 * self.lattice.w[2]*self.rho0_ones[:,-1]*(jnp.tensordot(self.lattice.c[:,2], u_bc[:, -1], axes=(-1, -1))/self.lattice.cs2))
            return f_i
        f_i = bounce_back_couette2D(f, f_prev)
        f_i = moving_wall_correction(f_i)
        return f_i

    def plot(self, f, it):
        rho, u = self.macro_vars(f)
        print(u[int(self.nx / 2), :, 0].mean())
        u_magnitude = jnp.linalg.norm(u, axis=-1, ord=2)
        plt.imshow(u[:,:,0].T, cmap='viridis')
        plt.imshow(u_magnitude.T, cmap='viridis')
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title("it:" + str(it) + "sum_rho:" + str(jnp.sum(rho)))
        plt.savefig(self.sav_dir + "/fig_2D_it" + str(it) + ".jpg")
        plt.clf()

def couette_analytical():
    y = jnp.arange(1, ny + 1) - 0.5
    ybottom = 0
    ytop = ny
    u_analytical = u_top_wall / ny*y
    return u_analytical

if __name__ == "__main__":
    time1 = time.time()
    nx = 180
    ny = 30
    nt = int(1e4)
    rho0 = 1
    tau = 1
    lattice = LatticeD2Q9()
    plot_every = 100
    u_bc = jnp.zeros((nx, ny, 2))
    u_top_wall = 0.1
    u_bc = u_bc.at[:, -1, 0].set(u_top_wall)
    kwargs = {
        'lattice': lattice,
        'tau': tau,
        'nx': nx,
        'ny': ny,
        'rho0': rho0,
        'plot_every': plot_every,
        'u_bc': u_bc,
    }
    sim = Couette(**kwargs)
    sim.run(nt)
    time2 = time.time()
    print(time2-time1)