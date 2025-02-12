import matplotlib.pyplot as plt
from src.lattice import LatticeD2Q9
from src.model import BGK
import jax.numpy as jnp
import time
"""
2D poiseuille flow driven by a gravity-like body force.
Vertical tube domain, no tilt.

             Periodic BC
        +--------------------+
        |                    |
        |                    |
        |                    |
        |         ||         |
No-slip |         ||         | No-slip
  BC    |         ||         |   BC
        |         ||         |
        |         \/         |
        |                    |
        |                    |
        |                    |
      (0,0)------------------+
             Periodic BC
"""
class Poiseuille(BGK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return "Poiseuille_gravity_force"

    def apply_bc(self, f, f_prev, force=None):
        def bb_vertical_tube2D(f_i):
            # bounce-back left wall
            # f_i = f_i.at[0, :, 5].set(f_prev[0, :, 7])
            # f_i = f_i.at[0, :, 1].set(f_prev[0, :, 3])
            # f_i = f_i.at[0, :, 8].set(f_prev[0, :, 6])
            f_i = f_i.at[0, :, self.lattice.right_indices].set(f_prev[0, :, self.lattice.opp_indices[self.lattice.right_indices]])
            # bounce-back right wall
            # f_i = f_i.at[-1, :, 6].set(f_prev[-1, :, 8])
            # f_i = f_i.at[-1, :, 3].set(f_prev[-1, :, 1])
            # f_i = f_i.at[-1, :, 7].set(f_prev[-1, :, 5])
            f_i = f_i.at[-1, :, self.lattice.left_indices].set(f_prev[-1, :, self.lattice.opp_indices[self.lattice.left_indices]])
            return f_i
        f = bb_vertical_tube2D(f)
        return f

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