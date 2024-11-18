import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import datetime
from functools import partial
from jax import jit

class LBM:
    def __init__(self, **kwargs):
        self.nx = kwargs.get("nx") #x dimension
        self.ny = kwargs.get("ny") #y dimension
        self.nz = kwargs.get("nz") #z dimension
        # self.nt = kwargs.get("nt") #time steps #make argument of run()
        self.rho0 = kwargs.get("rho0") #density for initialisation
        self.boundary_conditions = kwargs.get("boundary_conditions")
        self.lattice = kwargs.get("lattice") #set lattice
        # self.plotter = kwargs.get("plotter") #set plotter

        # defining the following lattice parameters isn't necessary, just refer using self.lattice.x
        # self.d = self.lattice.d #space dimensions number from lattice
        # self.q = self.lattice.q #velocity dimensions number from lattice
        # self.c = self.lattice.c #lattice velocities corresponding to lattice type
        # self.w = self.lattice.w #weights corresponding to lattice velocities

        self.dimensions = [self.nx or 0, self.ny or 0, self.nz or 0]
        self.dimensions = self.dimensions[:self.lattice.d]
        self.rho_dimension = tuple(self.dimensions)
        self.u_dimension = (*self.dimensions, self.lattice.d)

        #TODO do not include this in main LBM function, find something better
        today = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        cwd = os.path.abspath(__file__)
        self.sav_dir = os.path.join(os.path.dirname(cwd), "../test", today)
        if not os.path.isdir(self.sav_dir):
            os.makedirs(self.sav_dir)

    def run(self, nt):
        """
        Runs the model
        1. Initialise simulation
            Initialise()
        2. iterate over nt
            update()
        3. store values
            #TODO
        """
        f = self.initialize()
        for it in range(nt):
            f, f_prev = self.update(f) #updates f
            if it % 100 == 0:
                self.plot(f, it)
        return f

    def initialize(self):
        """
        calculates initial state from equilibrium where initial macroscopic values are:
            1. Velocities = 0
            2. Densities = rho0
        For entire domain
        """
        u = jnp.zeros(self.u_dimension)
        rho = self.rho0 * jnp.ones(self.rho_dimension)
        f = self.equilibrium(rho, u)
        return f

    @partial(jax.jit, static_argnums=0)
    def update(self, f_prev):
        """
        updates discrete velocities
            1. Collision step
                collision(f)
            2. Apply Boundary conditions
                (on simulation level)
            3. Stream discrete velocities
                stream(f)
        """
        f_post_col = self.collision(f_prev)
        f_post_col = self.apply_pre_bc(f_post_col, f_prev)
        f_post_col = self.stream(f_post_col)
        f_post_col = self.apply_bc(f_post_col)
        return f_post_col, f_prev

    @partial(jit, static_argnums=0, inline=True)
    def macro_vars(self, f):
        rho = jnp.sum(f, axis=-1)
        # u = jnp.dot(f, self.lattice.c.T) / rho[..., jnp.newaxis]
        # u = jnp.dot(f, self.lattice.c.T)
        ux = jnp.sum(f[:,:,[1,5,8]], axis=2) - jnp.sum(f[:,:,[3,6,7]], axis=2)
        uy = jnp.sum(f[:,:,[2,5,6]], axis=2) - jnp.sum(f[:,:,[4,7,8]], axis=2)
        u = jnp.stack([ux, uy], axis=-1)
        return rho, u

    def write_disk(self):
        """
        store macroscopic values in array for use of outside visualisation software.
        Writing an XML of some kind for VTK to be implemented in ParaView
        """
        #TODO
        pass

    def collision(self, f):
        """
        --Specified in model class--
        Applies collision
        """
        pass

    def apply_bc(self, f):
        """
        --Specified in simulation class--
        Applies boundary conditions after streaming
        """
        pass

    def apply_pre_bc(self, f, f_prev):
        """
        --Specified in simulation class--
        Applies boundary conditions before streaming
        if not defined, returns post-collision
        """
        return f

    @partial(jit, static_argnums=(0,))
    def stream(self, f):
        """
        Streams discrete velocities to neighbors.
        """
        def stream_i(f_i, c):
            if self.lattice.d == 1:
                return jnp.roll(f_i, (c[0]), axis=0)
            if self.lattice.d == 2:
                return jnp.roll(f_i, (c[0], c[1]), axis=(0, 1))
            if self.lattice.d == 3:
                return jnp.roll(f_i, (c[0], c[1], c[2]), axis=(0, 1, 2))
        return jax.vmap(stream_i, in_axes=(-1, 0), out_axes=-1)(f, self.lattice.c.T)

    @partial(jax.jit, static_argnums=0)
    def equilibrium(self, rho, u):
        # Scheme from LBM book, linear equilibrium with incompressible model from sample code
        # Calculate the dot product of u and c
        uc_dot = u[:, :, 0][:, :, jnp.newaxis] * self.lattice.c[0, :] + u[:, :, 1][:, :, jnp.newaxis] * self.lattice.c[1, :]
        # Multiply by 3 and add rho
        f_eq = self.lattice.w * (rho[:, :, jnp.newaxis] + 3 * uc_dot)
        return f_eq

    def plot(self, f, it):
        #TODO Has to be a better way to visualise this data
        rho, u = self.macro_vars(f)

        u_magnitude = jnp.linalg.norm(u, axis=-1, ord=2)
        # print(u_magnitude.shape)
        plt.imshow(u[:,:,0].T, cmap='viridis')
        # plt.imshow(u_magnitude.T, cmap='viridis')
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title("it:" + str(it) + "sum_rho:" + str(jnp.sum(rho)))
        plt.savefig(self.sav_dir + "/fig_13_it" + str(it) + ".jpg")
        plt.clf()

