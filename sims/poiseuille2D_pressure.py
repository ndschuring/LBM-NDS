import matplotlib.pyplot as plt
from src.lattice import LatticeD2Q9
from src.model import BGK
import jax.numpy as jnp
import jax
import time

#TODO make this thing
class Poiseuille(BGK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return "Poiseuille_pressure"

    def apply_bc(self, f, f_prev):
        def bb_tube2d(f_i):
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

    # def force_term(self, f):
    #     force_x = jnp.zeros((self.nx, self.ny))
    #     force_x = force_x.at[0, :].set(1)
    #     return jnp.stack((force_x, jnp.zeros_like(force_x)), axis=-1)
    #     # return jnp.stack((rho, rho), axis=-1)

    # def nebb_poiseuille(self, f, f_prev, force, angle=0, c=1):
    #     rho, u = self.macro_vars(f)
    #     N_x =
    #     N_y =
    #     # left wall inlet
    #     f = f.at[0, :, 1].set(f[0, :, 3]+((2*rho[0, :]*u[0,:,0])/(3*c))-(force[0,:,0]/6*c))
    #     f = f.at[0, :, 5].set(f[0, :, 7]-1/2*(f[0, :, 2]+f[0, :, 4]))
    #     f = f.at[0, :, 8].set(f[0, :, 6]+1/2*(f[0, :, 2]+f[0, :, 4])+)

    def plot(self, f, it):
        rho, u = self.macro_vars(f)
        u_magnitude = jnp.linalg.norm(u, axis=-1, ord=2)
        plt.imshow(u[:,:,0].T, cmap='viridis')
        plt.imshow(u_magnitude.T, cmap='viridis')
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title("it:" + str(it) + "sum_rho:" + str(jnp.sum(rho)))
        plt.savefig(self.sav_dir + "/fig_2D_it" + str(it) + ".jpg")
        plt.clf()
        # plt.plot(self.x, u[:, int(self.nx / 2), 1], label="Velocity Profile")
        # plt.legend()
        # plt.savefig(self.sav_dir + "/fig_1D_it" + str(it) + ".jpg")
        # plt.clf()

if __name__ == "__main__":
    time1 = time.time()
    nx = 180
    ny = 30
    # nt = int(1e4)
    nt = int(1e2)
    rho0 = 1
    tau = 1
    lattice = LatticeD2Q9()
    plot_every = 1
    u_in = 0.1 #inlet velocity

    kwargs = {
        'lattice': lattice,
        'tau': tau,
        'nx': nx,
        'ny': ny,
        'rho0': rho0,
        'plot_every': plot_every,
    }
    simPoiseuille = Poiseuille(**kwargs)
    simPoiseuille.run(nt)
    time2 = time.time()
    print(time2-time1)