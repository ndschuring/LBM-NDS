import matplotlib.pyplot as plt
from src.lattice import LatticeD2Q9
import jax.numpy as jnp
from src.model import BGKMulti
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

class Couette(BGKMulti):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.u_bc = kwargs.get('u_bc')

    def __str__(self):
        return "Couette_Flow_multi"

    def apply_bc(self, f, f_prev, force=None, **kwargs):
        g_tag = kwargs.get('g_tag', False)
        rho, u = self.macro_vars(f)
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
            # f_i = f_i.at[:, -1, 7].add( - 2 * self.lattice.w[5]*self.rho0_ones[:,-1]*(jnp.tensordot(self.lattice.c[:,5], u_bc[:, -1], axes=(-1, -1))/self.lattice.cs2))
            # f_i = f_i.at[:, -1, 4].add( - 2 * self.lattice.w[2]*self.rho0_ones[:,-1]*(jnp.tensordot(self.lattice.c[:,2], u_bc[:, -1], axes=(-1, -1))/self.lattice.cs2))
            # f_i = f_i.at[:, -1, 8].add( - 2 * self.lattice.w[6]*self.rho0_ones[:,-1]*(jnp.tensordot(self.lattice.c[:,6], u_bc[:, -1], axes=(-1, -1))/self.lattice.cs2))
            f_i = f_i.at[:, -1, 7].add( - 2 * self.lattice.w[5]*rho[:,-1]*(jnp.tensordot(self.lattice.c[:,5], u_bc[:, -1], axes=(-1, -1))/self.lattice.cs2))
            f_i = f_i.at[:, -1, 4].add( - 2 * self.lattice.w[2]*rho[:,-1]*(jnp.tensordot(self.lattice.c[:,2], u_bc[:, -1], axes=(-1, -1))/self.lattice.cs2))
            f_i = f_i.at[:, -1, 8].add( - 2 * self.lattice.w[6]*rho[:,-1]*(jnp.tensordot(self.lattice.c[:,6], u_bc[:, -1], axes=(-1, -1))/self.lattice.cs2))
            return f_i

        f_i = bounce_back_couette2D(f, f_prev)
        # if not g_tag:
        f_i = moving_wall_correction(f_i)
        return f_i

    def plot(self, f, it, **kwargs):
        g = kwargs.get('g')
        rho, u = self.macro_vars(f)
        phi, _ = self.macro_vars(g)
        print(u[int(self.nx / 2), :, 0].mean())
        u_magnitude = jnp.linalg.norm(u, axis=-1, ord=2)
        # plt.imshow(u[:,:,0].T, cmap='viridis')
        plt.imshow(u_magnitude.T, cmap='viridis')
        # u_masked = jnp.where(wall_mask, 0, u_magnitude)
        # plt.imshow(u_masked.T, cmap='viridis')
        plt.gca().invert_yaxis()
        plt.title("it:" + str(it) + "sum_rho:" + str(jnp.sum(rho[1:-1, 1:-1])))
        plt.colorbar(label="velocity magnitude")
        plt.xlabel("x [lattice units]")
        plt.ylabel("y [lattice units]")
        plt.savefig(self.sav_dir + "/fig_2D_it" + str(it) + ".jpg", dpi=100)
        plt.clf()
        plt.imshow(phi.T, cmap='viridis', vmin=-1, vmax=1)
        # phi_masked = jnp.where(wall_mask, 0, phi)
        # plt.imshow(phi_masked.T, cmap='viridis')
        plt.gca().invert_yaxis()
        plt.colorbar(label="Order Parameter")
        plt.xlabel("x [lattice units]")
        plt.ylabel("y [lattice units]")
        plt.title("it:"+str(it)+"Order parameter phi")
        plt.savefig(self.sav_dir+"/fig_2D_phi_it"+str(it)+".jpg", dpi=100)
        plt.clf()

def couette_analytical():
    y = jnp.arange(1, ny + 1) - 0.5
    ybottom = 0
    ytop = ny
    u_analytical = u_top_wall / ny*y
    return u_analytical

if __name__ == "__main__":
    time1 = time.time()
    nx = 16
    ny = 16
    nt = int(3e5)
    # nt = int(2e2)
    rho0 = 1
    tau = 1
    lattice = LatticeD2Q9()
    plot_every = 1000
    # initialise u_bc, a matrix mask specifying which velocities need to be enforced at certain coordinates
    u_bc = jnp.zeros((nx, ny, 2))
    u_top_wall = 0.1
    u_bc = u_bc.at[:, -1, 0].set(u_top_wall)
    # wall_mask = jnp.zeros((nx, ny), dtype=bool)
    # wall_mask = wall_mask.at[0,:].set(True).at[-1,:].set(True).at[:,0].set(True).at[:,-1].set(True)
    phi_init = jnp.zeros((nx, ny)).at[:, :int(ny/2)].set(1)
    phi_init = phi_init.at[:, int(ny/2):].set(-1)
    # phi_init = jnp.zeros_like(phi_init).at[:,:].set(1e-10)
    # phi_init = phi_init.at[3, 5].set(-1)
    # print(phi_init)
    gamma = 1
    param_A = param_B = -4e-4
    kappa = 4.5*jnp.abs(param_A)
    kappa = 2.5e-2
    tau_phi = 1
    kwargs = {
        'lattice': lattice,
        'tau': tau,
        'nx': nx,
        'ny': ny,
        'rho0': rho0,
        'plot_every': plot_every,
        'u_bc': u_bc,
        'gamma': gamma,
        'param_A': param_A,
        'param_B': param_B,
        'kappa': kappa,
        'tau_phi': tau_phi,
        'phi_init': phi_init,
    }
    sim = Couette(**kwargs)
    sim.run(nt)
    time2 = time.time()
    print(time2-time1)
# ----------------------------------------------------------------------------------------------------------------------
# import matplotlib.pyplot as plt
# from src.lattice import LatticeD2Q9
# import jax.numpy as jnp
# from src.model import BGKMulti
# import time
# """
# Couette Flow
#                 Generalised
#                 moving Wall BC (Krüger et al.)
#         +-------------------------------------------+
#         |                  ------>                  |
# Periodic|                                           |Periodic
#     BC  |                                           |   BC
#         |                                           |
#         |                                           |
#       (0,0)-----------------------------------------+
#                         No slip BC
# """
#
# class Couette(BGKMulti):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.u_bc = kwargs.get('u_bc')
#
#     def __str__(self):
#         return "Couette_Flow_multi"
#
#     def apply_bc(self, f, f_prev, force=None):
#         def bounce_back_couette2D(f_i, f_prev):
#             # Bounce-back top wall
#             f_i = f_i.at[:, -2, 7].set(f_prev[:, -2, 5])# - 2 * self.lattice.w[5]*self.rho0_ones[:,-1]*(jnp.tensordot(self.lattice.c[:,5], u_bc[:, -2], axes=(-1, -1))/self.lattice.cs2))
#             f_i = f_i.at[:, -2, 4].set(f_prev[:, -2, 2])# - 2 * self.lattice.w[2]*self.rho0_ones[:,-1]*(jnp.tensordot(self.lattice.c[:,2], u_bc[:, -2], axes=(-1, -1))/self.lattice.cs2))
#             f_i = f_i.at[:, -2, 8].set(f_prev[:, -2, 6])# - 2 * self.lattice.w[6]*self.rho0_ones[:,-1]*(jnp.tensordot(self.lattice.c[:, 6], u_bc[:, -2], axes=(-1, -1))/self.lattice.cs2))
#             # Bounce-back bottom wall
#             f_i = f_i.at[:,  1, 6].set(f_prev[:,  1, 8])
#             f_i = f_i.at[:,  1, 2].set(f_prev[:,  1, 4])
#             f_i = f_i.at[:,  1, 5].set(f_prev[:,  1, 7])
#             return f_i
#         def periodic_horizontal(f_i):
#             # Manual periodic boundary conditions to
#             right_new = jnp.copy(f_i[1, 1:-1, self.lattice.left_indices])
#             left_new =jnp.copy(f_i[-2, 1:-1, self.lattice.right_indices])
#             f_i = f_i.at[1, 1:-1, self.lattice.right_indices].set(left_new)
#             f_i = f_i.at[-2, 1:-1, self.lattice.left_indices].set(right_new)
#             return f_i
#         def moving_wall_correction(f_i):
#             # top wall
#             f_i = f_i.at[:, -2, 7].add( - 2 * self.lattice.w[5]*self.rho0_ones[:,-1]*(jnp.tensordot(self.lattice.c[:,5], u_bc[:, -2], axes=(-1, -1))/self.lattice.cs2))
#             f_i = f_i.at[:, -2, 4].add( - 2 * self.lattice.w[2]*self.rho0_ones[:,-1]*(jnp.tensordot(self.lattice.c[:,2], u_bc[:, -2], axes=(-1, -1))/self.lattice.cs2))
#             f_i = f_i.at[:, -2, 8].add( - 2 * self.lattice.w[6]*self.rho0_ones[:,-1]*(jnp.tensordot(self.lattice.c[:,6], u_bc[:, -2], axes=(-1, -1))/self.lattice.cs2))
#             return f_i
#         f_i = bounce_back_couette2D(f, f_prev)
#         f_i = periodic_horizontal(f_i)
#         # f_i = moving_wall_correction(f_i)
#         return f_i
#
#     def plot(self, f, it, **kwargs):
#         g = kwargs.get('g')
#         rho, u = self.macro_vars(f)
#         phi, _ = self.macro_vars(g)
#         print(u[int(self.nx / 2), :, 0].mean())
#         u_magnitude = jnp.linalg.norm(u, axis=-1, ord=2)
#         # plt.imshow(u[:,:,0].T, cmap='viridis')
#         plt.imshow(u_magnitude.T, cmap='viridis')
#         u_masked = jnp.where(wall_mask, 0, u_magnitude)
#         plt.imshow(u_masked.T, cmap='viridis')
#         plt.gca().invert_yaxis()
#         plt.colorbar()
#         plt.title("it:" + str(it) + "sum_rho:" + str(jnp.sum(rho[1:-1, 1:-1])))
#         plt.savefig(self.sav_dir + "/fig_2D_it" + str(it) + ".jpg", dpi=100)
#         plt.clf()
#         phi_masked = jnp.where(wall_mask, 0, phi)
#         plt.imshow(phi_masked.T, cmap='viridis')
#         plt.gca().invert_yaxis()
#         plt.colorbar()
#         plt.title("it:"+str(it)+"Order parameter phi")
#         plt.savefig(self.sav_dir+"/fig_2D_phi_it"+str(it)+".jpg", dpi=100)
#         plt.clf()
#
# def couette_analytical():
#     y = jnp.arange(1, ny + 1) - 0.5
#     ybottom = 0
#     ytop = ny
#     u_analytical = u_top_wall / ny*y
#     return u_analytical
#
# if __name__ == "__main__":
#     time1 = time.time()
#     nx = 180
#     ny = 30
#     nt = int(1e4)
#     rho0 = 1
#     tau = 1
#     lattice = LatticeD2Q9()
#     plot_every = 100
#     # initialise u_bc, a matrix mask specifying which velocities need to be enforced at certain coordinates
#     u_bc = jnp.zeros((nx, ny, 2))
#     u_top_wall = 0.1
#     u_bc = u_bc.at[1:-1, -2, 0].set(u_top_wall)
#     wall_mask = jnp.zeros((nx, ny), dtype=bool)
#     wall_mask = wall_mask.at[0,:].set(True).at[-1,:].set(True).at[:,0].set(True).at[:,-1].set(True)
#     phi_init = jnp.zeros((nx, ny)).at[:, :int(ny/2)].set(1)
#     phi_init = phi_init.at[:, int(ny/2):].set(-1)
#     # print(phi_init)
#     gamma = 1
#     param_A = param_B = -4e-4
#     kappa = 4.5*jnp.abs(param_A)
#     tau_phi = 1
#     kwargs = {
#         'lattice': lattice,
#         'tau': tau,
#         'nx': nx,
#         'ny': ny,
#         'rho0': rho0,
#         'plot_every': plot_every,
#         'u_bc': u_bc,
#         'gamma': gamma,
#         'param_A': param_A,
#         'param_B': param_B,
#         'kappa': kappa,
#         'tau_phi': tau_phi,
#         'phi_init': phi_init,
#     }
#     sim = Couette(**kwargs)
#     sim.run(nt)
#     time2 = time.time()
#     print(time2-time1)