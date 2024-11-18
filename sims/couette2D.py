from src.boundary_conditions import *
from src.lattice import LatticeD2Q9
from src.model import BGK
import time
"""
Couette Flow

                Moving Wall BC (Krüger et al.)
        +-------------------------------------------+
        |                  ------>                  |
Periodic|                                           |Periodic
    BC  |                                           |   BC
        |                                           |
        |                                           |
        +-------------------------------------------+
                        No slip BC
"""



class couette(BGK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply_bc(self, f):
        return bounce_back_couette2D(f)

if __name__ == "__main__":
    time1 = time.time()
    nx = 180
    ny = 30
    nt = 1000
    rho0 = 1
    tau = 1
    lattice = LatticeD2Q9()
    kwargs = {
        'lattice': lattice,
        'tau': tau,
        'nx': nx,
        'ny': ny,
        'rho0': rho0,
    }
    sim = couette(**kwargs)
    sim.run(nt)
    time2 = time.time()
    print(time2-time1)