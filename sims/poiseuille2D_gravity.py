import matplotlib.pyplot as plt
from src.lattice import LatticeD2Q9
from src.model import BGK
import jax.numpy as jnp
import time

class Poiseuille(BGK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply_bc(self, f, f_prev):
        def bb_vertical_tube2D(f_i):
            # bounce-back left wall
            f_i = f_i.at[0, :, 5].set(f_i[0, -1, 7])
            f_i = f_i.at[0, :, 1].set(f_i[0, -1, 3])
            f_i = f_i.at[0, :, 8].set(f_i[0, -1, 6])
            # bounce-back right wall
            f_i = f_i.at[-1, :, 6].set(f_i[-1, 0, 8])
            f_i = f_i.at[-1, :, 3].set(f_i[-1, 0, 1])
            f_i = f_i.at[-1, :, 7].set(f_i[-1, 0, 5])
            return f_i
        f = bb_vertical_tube2D(f)
        return f

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
        plt.plot(self.x, u[:, int(self.nx / 2), 1], label="Velocity Profile")
        plt.legend()
        plt.savefig(self.sav_dir + "/fig_1D_it" + str(it) + ".jpg")
        plt.clf()


if __name__ == "__main__":
    time1 = time.time()
    nx = 30
    ny = 180
    nt = int(1e4)
    # nt = int(1e3)
    rho0 = 1
    tau = 1
    lattice = LatticeD2Q9()
    plot_every = 100
    g_set = 0.000003
    tilt_angle = 0.0

    kwargs = {
        'lattice': lattice,
        'tau': tau,
        'nx': nx,
        'ny': ny,
        'rho0': rho0,
        'plot_every': plot_every,
        'g_set': g_set,
        'tilt_angle': tilt_angle,
    }
    simPoiseuille = Poiseuille(**kwargs)
    simPoiseuille.run(nt)
    time2 = time.time()
    print(time2-time1)