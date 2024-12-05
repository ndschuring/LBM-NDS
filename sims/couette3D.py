import matplotlib.pyplot as plt
from src.lattice import LatticeD3Q27
import jax.numpy as jnp
from src.model import BGK
import time
"""
Couette Flow
            ____________________________________________
           /       Generalised                         /|
          /        moving wall BC (Krüger et al.)     / |
         /           >-------------------->          /  |
        /___________________________________________/   | Periodic
        |                                           |   |   BC
Periodic|                                           |   | 
    BC  |                                           |   | 
        |                                           |  /
        |                                           | /
        |___________________________________________|/
                        No slip BC
"""



class Couette(BGK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return "Couette_Flow_3D"

    def apply_bc(self, f, f_prev, force=None):
        pass

if __name__ == "__main__":
    time1 = time.time()
    nx = 180
    ny = 30
    nz = 30
    nt = int(1e4)
    rho0 = 1
    tau = 1
    lattice = LatticeD3Q27()
    plot_every = 100
    kwargs = {
        'lattice': lattice,
        'tau': tau,
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'rho0': rho0,
        'plot_every': plot_every,
    }
    sim = Couette(**kwargs)
    sim.run(nt)
    time2 = time.time()
    print(time2-time1)