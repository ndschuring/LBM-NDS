from src.lattice import LatticeD2Q9
from src.model import BGK
import time
"""
Couette Flow

                Moving Wall BC (Krüger et al.)
                lbm-principles code
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

    def __str__(self):
        return "Couette_Flow"

    def apply_bc(self, f, f_prev):
        def bounce_back_couette2D(f_i, f_prev):
            # Bounce-back top wall with hard-coded correction term
            u_max = 0.1
            f_i = f_i.at[:, -1, 7].set(f_prev[:, -1, 5] - 1 / 6 * u_max)
            f_i = f_i.at[:, -1, 4].set(f_prev[:, -1, 2])
            f_i = f_i.at[:, -1, 8].set(f_prev[:, -1, 6] + 1 / 6 * u_max)
            # Bounce-back bottom wall
            f_i = f_i.at[:, 0, 6].set(f_prev[:, 0, 8])
            f_i = f_i.at[:, 0, 2].set(f_prev[:, 0, 4])
            f_i = f_i.at[:, 0, 5].set(f_prev[:, 0, 7])
            return f_i
        return bounce_back_couette2D(f, f_prev)

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
    # Set kwargs
    kwargs = {
        'lattice': lattice,
        'tau': tau,
        'nx': nx,
        'ny': ny,
        'rho0': rho0,
        'plot_every': plot_every,
    }
    # Create simulation and run
    sim = Couette(**kwargs)
    sim.run(nt)
    time2 = time.time()
    print(time2-time1)