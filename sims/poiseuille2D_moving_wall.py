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
        return "Poiseuille_moving_wall"

    def apply_bc(self, f, f_prev, force=None):
        def bb_tube2d(f_i):
            rho, u = self.macro_vars(f_prev)
            rho = jnp.ones_like(rho)
            # bounce-back left wall (moving wall inlet velocity)
            # equation (5.26)
            f_i = f_i.at[0, :, 5].set(f_prev[0, :, 6] - 2 * self.lattice.w[6]*self.rho0_ones[0, :]*(jnp.tensordot(self.lattice.c[:,6], u_bc[0, :], axes=(-1, -1))/self.lattice.cs2))
            f_i = f_i.at[0, :, 1].set(f_prev[0, :, 3] - 2 * self.lattice.w[3]*self.rho0_ones[0, :]*(jnp.tensordot(self.lattice.c[:,3], u_bc[0, :], axes=(-1, -1))/self.lattice.cs2))
            f_i = f_i.at[0, :, 8].set(f_prev[0, :, 7] - 2 * self.lattice.w[7]*self.rho0_ones[0, :]*(jnp.tensordot(self.lattice.c[:,7], u_bc[0, :], axes=(-1, -1))/self.lattice.cs2))
            # anti-bounce-back pressure right wall
            # equation (5.53)
            # u_outlet = u[-1, :, :] + 0.5 * (u[-1, :, :] - u[-2, :, :])
            u_outlet = u[-1, :, :]
            # f_i = f_i.at[-1, :, 6].set(-f_prev[-1, :, 5] + 2 * self.lattice.w[5]*self.rho0_ones[-1, :]*(1+(jnp.tensordot(self.lattice.c[:,5], u_outlet, axes=(-1, -1))**2)/(2*self.lattice.cs2**2))-(jnp.einsum("ij,ij->i",u_outlet,u_outlet))/(2*self.lattice.cs2))
            # f_i = f_i.at[-1, :, 3].set(-f_prev[-1, :, 1] + 2 * self.lattice.w[1]*self.rho0_ones[-1, :]*(1+(jnp.tensordot(self.lattice.c[:,1], u_outlet, axes=(-1, -1))**2)/(2*self.lattice.cs2**2))-(jnp.einsum("ij,ij->i",u_outlet,u_outlet))/(2*self.lattice.cs2))
            # f_i = f_i.at[-1, :, 7].set(-f_prev[-1, :, 8] + 2 * self.lattice.w[8]*self.rho0_ones[-1, :]*(1+(jnp.tensordot(self.lattice.c[:,8], u_outlet, axes=(-1, -1))**2)/(2*self.lattice.cs2**2))-(jnp.einsum("ij,ij->i",u_outlet,u_outlet))/(2*self.lattice.cs2))
            f_i = f_i.at[-1, :, 6].set(-f_prev[-1, :, 5] + 2 * self.lattice.w[5] * self.rho0_ones[-1, :] * (
                        1 + (jnp.tensordot(self.lattice.c[:, 5], u_outlet, axes=(-1, -1)) ** 2) / (
                            2 * self.lattice.cs2 ** 2)))
            f_i = f_i.at[-1, :, 3].set(-f_prev[-1, :, 1] + 2 * self.lattice.w[1] * self.rho0_ones[-1, :] * (
                        1 + (jnp.tensordot(self.lattice.c[:, 1], u_outlet, axes=(-1, -1)) ** 2) / (
                            2 * self.lattice.cs2 ** 2)))
            f_i = f_i.at[-1, :, 7].set(-f_prev[-1, :, 8] + 2 * self.lattice.w[8] * self.rho0_ones[-1, :] * (
                        1 + (jnp.tensordot(self.lattice.c[:, 8], u_outlet, axes=(-1, -1)) ** 2) / (
                            2 * self.lattice.cs2 ** 2)))
            # bounce-back top wall
            f_i = f_i.at[:, -1, 7].set(f_prev[:, -1, 5])
            f_i = f_i.at[:, -1, 4].set(f_prev[:, -1, 2])
            f_i = f_i.at[:, -1, 8].set(f_prev[:, -1, 6])
            # bounce-back bottom wall
            f_i = f_i.at[:, 0, 5].set(f_prev[:, 0, 7])
            f_i = f_i.at[:, 0, 2].set(f_prev[:, 0, 4])
            f_i = f_i.at[:, 0, 6].set(f_prev[:, 0, 8])
            return f_i
        f = bb_tube2d(f)
        return f

    def plot(self, f, it):
        rho, u = self.macro_vars(f)
        # print(u[1, :, 0].mean())
        # print(u[int(self.nx / 2), :, 0].mean())
        u_magnitude = jnp.linalg.norm(u, axis=-1, ord=2)
        # plt.imshow(u[:,:,0].T, cmap='viridis')
        plt.imshow(u_magnitude.T, cmap='viridis')
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title("it:" + str(it) + "sum_rho:" + str(jnp.sum(rho)))
        plt.savefig(self.sav_dir + "/fig_2D_it" + str(it) + ".jpg", dpi=150)
        plt.clf()
        # plt.plot(self.x, u[:, int(self.nx / 2), 1], label="Velocity Profile")
        # plt.legend()
        # plt.savefig(self.sav_dir + "/fig_1D_it" + str(it) + ".jpg")
        # plt.clf()

def poiseuille_analytical():
    y = jnp.arange(1, ny + 1) - 0.5
    ybottom = 0
    ytop = ny
    u_analytical = -4 * u_in / (ny ** 2) * (y - ybottom) * (y - ytop)
    return u_analytical

if __name__ == "__main__":
    time1 = time.time()
    nx = 180
    ny = 30
    nt = int(1e4)
    # nt = int(4e2)
    rho0 = 1
    tau = 1
    lattice = LatticeD2Q9()
    plot_every = 100
    u_in = 0.1 #inlet velocity
    u_bc = jnp.zeros((nx, ny, 2))
    u_inlet = jnp.ones((nx, ny)) * u_in
    u_outlet = jnp.ones((nx, ny)) * u_in
    # u_inlet = poiseuille_analytical()
    # u_outlet = poiseuille_analytical()
    u_bc = u_bc.at[0, 1:-1, 0].set(u_inlet[0, 1:-1])
    u_bc = u_bc.at[-1, 1:-1, 0].set(u_outlet[-1, 1:-1])
    # u_bc = u_bc.at[0, :, 0].set(u_inlet.T)
    # u_bc = u_bc.at[-1, :, 0].set(u_outlet.T)
    kwargs = {
        'lattice': lattice,
        'tau': tau,
        'nx': nx,
        'ny': ny,
        'rho0': rho0,
        'plot_every': plot_every,
        'debug': (True, 299),
    }
    simPoiseuille = Poiseuille(**kwargs)
    simPoiseuille.run(nt)
    time2 = time.time()
    print(time2-time1)

