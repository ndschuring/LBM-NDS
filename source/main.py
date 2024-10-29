"""
Code by Sacha Szkudlarek
"""
import argparse
import datetime
import os

import jax
import matplotlib.pyplot as plt

from source.func.LBMfunc import *
from source.func.MacroscopicVariables import *
from source.func.BoundaryConditions import *
from init import *

# Ensures that every run is saved in a separate directory
today = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
cwd = os.path.abspath(__file__)
sav_dir = os.path.join(os.path.dirname(cwd) + "/test/", today)
if not os.path.isdir(sav_dir):
    os.makedirs(sav_dir)


def lbm():
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'cpu')

    @jax.jit
    def update(f_i_prev):
        """
        main update function

        Args:
            f_i_prev (jnp.array): discrete velocities of previous iteration

        Returns:
            f_streamed (jnp.array): discrete velocities of next iteration
        """
        # macroscopic variables
        rho_prev = get_density(f_i_prev)
        force_prev = force_term(rho_prev)
        u_prev = get_velocity(f_i_prev, force_prev, lattice_velocities) #lattice velocities is array with velocity directions, THIS is where we can specify different discretisations!
        # update procedure
        f_eq = equilibrium(rho_prev, u_prev)
        source = source_term(u_prev, force_prev)
        f_col = collision_bgk(f_i_prev, f_eq, source)
        f_bc = bounce_back(f_col)
        f_streamed = streaming(f_bc)
        return f_streamed

    rho, u = init()
    f_prev = equilibrium(rho, u)
    plt.figure()

    for it in range(nt):
        f_next = update(f_prev)
        f_prev = f_next

        if it % plot_every == 0 and it > skip_it:
            rho_ = get_density(f_next)
            plt.imshow(rho_.T, cmap='viridis')
            plt.gca().invert_yaxis()
            plt.colorbar()
            plt.title("it:" + str(it) + "sum_rho:" + str(jnp.sum(rho_)))
            plt.savefig(sav_dir + "/fig_13_it" + str(it) + ".jpg")
            plt.clf()


def main():
    lbm()


if __name__ == "__main__":
    main()
