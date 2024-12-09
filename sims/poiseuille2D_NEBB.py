import matplotlib.pyplot as plt
from src.lattice import LatticeD2Q9
from src.model import BGK
import jax.numpy as jnp
import jax
import time
"""
2D poiseuille flow driven by non-equilibrium bounce-back at all walls, with enforced velocity at inlet and outlet.
"""
class Poiseuille(BGK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return "Poiseuille_NEBB"

    def apply_bc(self, f, f_prev, force=None):
        rho, u = self.macro_vars(f_prev)
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
        # def nebb_inlet(f_i):
        # # WORKS
        #     Ny = (1/2)*(f_i[0, :, 4]-f_i[0, :, 2]) + (1/3)*rho[0, :]*u_bc[0, :, 1]
        #     f_i = f_i.at[0, :, 1].set(f_i[0, :, 3]+(2/3)*rho[0, :]*u_bc[0, :, 0])
        #     f_i = f_i.at[0, :, 5].set(f_i[0, :, 7]+(1/6)*rho[0, :]*(u_bc[0, :, 0]+u_bc[0, :, 1])+Ny)
        #     f_i = f_i.at[0, :, 8].set(f_i[0, :, 6]-(1/6)*rho[0, :]*(-u_bc[0, :, 0]+u_bc[0, :, 1])-Ny)
        #     return f_i
        # def nebb_outlet(f_i):
        #     Ny = (1/2)*(f_i[-1, :, 4]-f_i[-1, :, 2]) + (1/3)*rho[-1, :]*u_bc[-1, :, 1]
        #     f_i = f_i.at[-1, :, 3].set(f_i[-1, :, 1]+(2/3)*rho[-1, :]*u_bc[-1, :, 0])
        #     f_i = f_i.at[-1, :, 7].set(f_i[-1, :, 5]+(1/6)*rho[-1, :]*(u_bc[-1, :, 0]+u_bc[-1, :, 1])+Ny)
        #     f_i = f_i.at[-1, :, 6].set(f_i[-1, :, 8]-(1/6)*rho[-1, :]*(-u_bc[-1, :, 0]+u_bc[-1, :, 1])-Ny)
        #     return f_i
        def nebb_inlet(f_i):
            # rho = (f_i[0,:,0]+f_i[0,:,2]+f_i[0,:,4]+2*(f_i[0,:,6]+f_i[0,:,3]+f_i[0,:,7]))/(1-u_bc[0,:,0])
            # rho = (f_i[:,:,0]+f_i[:,:,2]+f_i[:,:,4]+2*(f_i[:,:,6]+f_i[:,:,3]+f_i[:,:,7]))/(1-u_bc[:,:,0])
            Ny = (1/2)*(f_i[0, :, 4]-f_i[0, :, 2]) + (1/3)*rho[0, :]*u_bc[0, :, 1]
            f_i = f_i.at[0, :, 1].set(f_i[0, :, 3] + self.f_equilibrium(rho, u_bc)[0, :, 1] - self.f_equilibrium(rho, u_bc)[0, :, 3])
            f_i = f_i.at[0, :, 5].set(f_i[0, :, 7] + self.f_equilibrium(rho, u_bc)[0, :, 5] - self.f_equilibrium(rho, u_bc)[0, :, 7] + Ny)
            f_i = f_i.at[0, :, 8].set(f_i[0, :, 6] + self.f_equilibrium(rho, u_bc)[0, :, 8] - self.f_equilibrium(rho, u_bc)[0, :, 6] - Ny)
            return f_i

        def nebb_outlet(f_i):
            # rho = (f_i[-1,:,0]+f_i[-1,:,2]+f_i[-1,:,4]+2*(f_i[-1,:,6]+f_i[-1,:,3]+f_i[-1,:,7]))/(1-u_bc[-1,:,0])
            # rho = (f_i[:,:,0]+f_i[:,:,2]+f_i[:,:,4]+2*(f_i[:,:,1]+f_i[:,:,5]+f_i[:,:,8]))/(1-u_bc[:,:,0])
            Ny = (1/2)*(f_i[-1, :, 4]-f_i[-1, :, 2]) + (1/3)*rho[-1, :]*u_bc[-1, :, 1]
            f_i = f_i.at[-1, :, 3].set(f_i[-1, :, 1] + self.f_equilibrium(rho, u_bc)[-1, :, 3] - self.f_equilibrium(rho, u_bc)[-1, :, 1])
            f_i = f_i.at[-1, :, 7].set(f_i[-1, :, 5] + self.f_equilibrium(rho, u_bc)[-1, :, 7] - self.f_equilibrium(rho, u_bc)[-1, :, 5] - Ny)
            f_i = f_i.at[-1, :, 6].set(f_i[-1, :, 8] + self.f_equilibrium(rho, u_bc)[-1, :, 6] - self.f_equilibrium(rho, u_bc)[-1, :, 8] + Ny)
            return f_i

        def nebb_top(f_i):
            Nx = -(1/2)*(f_i[:, -1, 1]-f_i[:, -1, 3]) + (1/3)*rho[:, -1]*u_bc[:, -1, 0]
            f_i = f_i.at[:, -1, 2].set(f_i[:, -1, 4] + self.f_equilibrium(rho, u_bc)[:, -1, 2] - self.f_equilibrium(rho, u_bc)[:, -1, 4])
            f_i = f_i.at[:, -1, 5].set(f_i[:, -1, 7] + self.f_equilibrium(rho, u_bc)[:, -1, 5] - self.f_equilibrium(rho, u_bc)[:, -1, 7] + Nx)
            f_i = f_i.at[:, -1, 6].set(f_i[:, -1, 8] + self.f_equilibrium(rho, u_bc)[:, -1, 6] - self.f_equilibrium(rho, u_bc)[:, -1, 8] - Nx)
            return f_i

        def nebb_bottom(f_i):
            Nx = -(1/2)*(f_i[:, 0, 1]-f_i[:, 0, 3]) + (1/3)*rho[:, 0]*u_bc[:, 0, 0]
            f_i = f_i.at[:, 0, 4].set(f_i[:, 0, 2] + self.f_equilibrium(rho, u_bc)[:, 0, 4] - self.f_equilibrium(rho, u_bc)[:, 0, 2])
            f_i = f_i.at[:, 0, 7].set(f_i[:, 0, 5] + self.f_equilibrium(rho, u_bc)[:, 0, 7] - self.f_equilibrium(rho, u_bc)[:, 0, 5] - Nx)
            f_i = f_i.at[:, 0, 8].set(f_i[:, 0, 6] + self.f_equilibrium(rho, u_bc)[:, 0, 8] - self.f_equilibrium(rho, u_bc)[:, 0, 6] + Nx)
            return f_i

        def nebb_corner_correction(f_i):
            f_i = f_i.at[ 0,  0, 0].set(rho[ 0,  0] - jnp.sum(f_i[ 0,  0, 1:], axis=-1))
            f_i = f_i.at[ 0, -1, 0].set(rho[ 0, -1] - jnp.sum(f_i[ 0, -1, 1:], axis=-1))
            f_i = f_i.at[-1,  0, 0].set(rho[-1,  0] - jnp.sum(f_i[-1,  0, 1:], axis=-1))
            f_i = f_i.at[-1, -1, 0].set(rho[-1, -1] - jnp.sum(f_i[-1, -1, 1:], axis=-1))
            return f_i
        f = bb_tube2d(f)
        f = nebb_inlet(f)
        f = nebb_outlet(f)
        # f = nebb_top(f)
        # f = nebb_bottom(f)
        # f = nebb_corner_correction(f)
        return f

    def plot(self, f, it):
        rho, u = self.macro_vars(f)
        print(u[int(self.nx / 2), :, 0].mean())
        u_magnitude = jnp.linalg.norm(u, axis=-1, ord=2)
        plt.imshow(u[:,:,0].T, cmap='viridis')
        # plt.imshow(u_magnitude.T, cmap='viridis')
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title("it:" + str(it) + "sum_rho:" + str(jnp.sum(rho)))
        plt.savefig(self.sav_dir + "/fig_2D_it" + str(it) + ".jpg")
        plt.clf()
        # plt.plot(self.y, u[4, :, 0], label="Velocity Profile nx=4")
        # plt.plot(self.y, u[40, :, 0], label="Velocity Profile nx=40")
        # plt.plot(self.y, u[int(self.nx / 2),:, 0], label="Velocity Profile nx=nx/2")
        # plt.plot(self.y, u[-40, :, 0], label="Velocity Profile nx=-40")
        # plt.plot(self.y, u[-4, :, 0], label="Velocity Profile nx=-4")
        # plt.plot(self.y, poiseuille_analytical(), label="Analytical")
        # plt.legend()
        # plt.savefig(self.sav_dir + "/fig_1D_it" + str(it) + ".jpg")
        # plt.clf()
        # plt.plot(self.y, jnp.abs(poiseuille_analytical() - u[int(self.nx / 2), :, 0]), label="error")
        # plt.savefig(self.sav_dir + "/fig_error_it" + str(it) + ".jpg")
        # plt.clf()

def poiseuille_analytical():
    y = jnp.arange(1, ny + 1) - 0.5
    # y = jnp.arange(0, ny + 1)
    ybottom = 0
    ytop = ny
    u_analytical = -4 * u_in / (ny ** 2) * (y - ybottom) * (y - ytop)
    return u_analytical

if __name__ == "__main__":
    time1 = time.time()
    nx = 180
    ny = 30
    nt = int(1e5)
    # nt = int(1e2)
    rho0 = 1
    tau = 1
    lattice = LatticeD2Q9()
    plot_every = 500
    u_in = 0.1 #inlet velocity
    u_bc = jnp.zeros((nx, ny, 2))
    u_inlet = jnp.ones((nx, ny)) * u_in
    u_outlet = jnp.ones((nx, ny)) * u_in
    # u_inlet = poiseuille_analytical()
    # u_outlet = poiseuille_analytical()
    u_bc = u_bc.at[0, 1:-1, 0].set(u_inlet[0,1:-1])
    u_bc = u_bc.at[-1, 1:-1, 0].set(u_outlet[-1, 1:-1])
    # u_bc = u_bc.at[0, 1:-1, 0].set(u_inlet[1:-1].T)
    # u_bc = u_bc.at[-1, 1:-1, 0].set(u_outlet[1:-1].T)
    # g_set = 0.000003
    # tilt_angle = 90
    print(poiseuille_analytical().mean())


    kwargs = {
        'lattice': lattice,
        'tau': tau,
        'nx': nx,
        'ny': ny,
        'rho0': rho0,
        'plot_every': plot_every,
        # 'g_set': g_set,
        # 'tilt_angle': tilt_angle,
    }
    simPoiseuille = Poiseuille(**kwargs)
    simPoiseuille.run(nt)
    time2 = time.time()
    print(time2-time1)