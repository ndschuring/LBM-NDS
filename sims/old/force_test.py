from src.lattice import LatticeD2Q9, LatticeD3Q27
from src.model import BGK
import jax.numpy as jnp
import time

class Tester(BGK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return "tester"

if __name__ == "__main__":
    time1 = time.time()
    nx = 15
    ny = 10
    nz = 7
    nt = int(1e4)
    rho0 = 1
    tau = 1
    lattice = LatticeD2Q9()
    plot_every = 100
    g_set = 0.001
    kwargs = {
        'lattice': lattice,
        'tau': tau,
        'nx': nx,
        'ny': ny,
        'rho0': rho0,
        'plot_every': plot_every,
        'g_set': g_set,
    }
    sim_D2Q9 = Tester(**kwargs)
    f_test_D2Q9 = sim_D2Q9.initialize()
    force_term_D2Q9 = sim_D2Q9.force_term(f_test_D2Q9)
    # source_old_D2Q9 = sim_D2Q9.source_term_(f_test_D2Q9, force_term_D2Q9)
    source_new_D2Q9 = sim_D2Q9.source_term(f_test_D2Q9, force_term_D2Q9)
    # equiv = jnp.array_equiv(source_old_D2Q9, source_new_D2Q9)
    # print(source_old_D2Q9.shape)
    print(source_new_D2Q9.shape)
    # print(equiv)

    lattice = LatticeD3Q27()
    kwargs = {
        'lattice': lattice,
        'tau': tau,
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'rho0': rho0,
        'plot_every': plot_every,
        'g_set': g_set,
    }
    sim_D3Q27 = Tester(**kwargs)
    f_test_D3Q27 = sim_D3Q27.initialize()
    force_term_D3Q27 = sim_D3Q27.force_term(f_test_D3Q27)
    # source_old_D3Q27 = sim_D3Q27.source_term_(f_test_D3Q27, force_term_D3Q27)
    source_new_D3Q27 = sim_D3Q27.source_term(f_test_D3Q27, force_term_D3Q27)
    # equiv = jnp.array_equiv(source_old_D3Q27, source_new_D3Q27)
    # print(equiv)
    # print(source_old_D3Q27.shape)
    print(source_new_D3Q27.shape)
    time2 = time.time()
    print(time2-time1)