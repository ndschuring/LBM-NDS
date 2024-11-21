import jax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import datetime
from functools import partial
from jax import jit

#    6   2   5
#      \ | /
#    3 - 0 - 1
#      / | \
#    7   4   8

class LBM:
    def __init__(self, **kwargs):
        self.nx = kwargs.get("nx") #x dimension
        self.ny = kwargs.get("ny") #y dimension
        self.nz = kwargs.get("nz") #z dimension
        # self.nt = kwargs.get("nt") #time steps #make argument of run()
        self.rho0 = kwargs.get("rho0") #density for initialisation
        self.boundary_conditions = kwargs.get("boundary_conditions")
        self.lattice = kwargs.get("lattice") #set lattice
        self.plot_every = kwargs.get("plot_every", 50)
        self.g_set = kwargs.get("g_set", 0)
        self.tilt_angle = kwargs.get("tilt_angle", 0)
        self.x = jnp.arange(1, self.nx+1) - 0.5
        self.y = jnp.arange(1, self.ny+1) - 0.5

        # defining the following lattice parameters isn't necessary, just refer using self.lattice.x
        # self.d = self.lattice.d #space dimensions number from lattice
        # self.q = self.lattice.q #velocity dimensions number from lattice
        # self.c = self.lattice.c #lattice velocities corresponding to lattice type
        # self.w = self.lattice.w #weights corresponding to lattice velocities

        self.dimensions = [self.nx or 0, self.ny or 0, self.nz or 0]
        self.dimensions = self.dimensions[:self.lattice.d]
        self.rho_dimension = tuple(self.dimensions)
        self.u_dimension = tuple((*self.dimensions, self.lattice.d))

        #TODO do not include this in main LBM function, find something better
        today = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
            if it % self.plot_every == 0:
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
        force_prev = self.force_term(f_prev)
        source_prev = self.source_term(f_prev, force_prev)
        f_post_col = self.collision(f_prev, source_prev)
        f_post_col = self.apply_pre_bc(f_post_col, f_prev)
        f_post_stream = self.stream(f_post_col)
        f_post_stream = self.apply_bc(f_post_stream, f_post_col)
        return f_post_stream, f_prev

    @partial(jit, static_argnums=0, inline=True)
    def macro_vars(self, f):
        rho = jnp.sum(f, axis=-1)
        # u = jnp.dot(f, self.lattice.c.T) / rho[..., jnp.newaxis] #<- does not work with poiseuille BC :(
        u = jnp.dot(f, self.lattice.c.T)
        return rho, u

    def write_disk(self):
        """
        store macroscopic values in array for use of outside visualisation software.
        Writing an XML of some kind for VTK to be implemented in ParaView
        """
        #TODO
        pass

    def collision(self, f, source):
        """
        --Specified in model class--
        Applies collision
        """
        pass

    def apply_bc(self, f, f_prev):
        """
        --Specified in simulation class--
        Applies boundary conditions after streaming
        """
        pass

    def force_term(self, f):
        rho, u = self.macro_vars(f)
        force_g = - rho * self.g_set
        force_parr = - force_g * jnp.sin(self.tilt_angle)
        force_perp = force_g * jnp.cos(self.tilt_angle)
        force_components = jnp.stack((force_parr, force_perp), axis=0)
        return force_components

    def apply_force(self, u, force):
        return u*0.5*force

    def source_term(self, u, force):
        ux, uy = u[:,:,0], u[:,:,1]
        fx, fy = force[0], force[1]
        cx, cy = self.lattice.c[0], self.lattice.c[1]
        source_ = jnp.zeros((self.nx, self.ny, 9))
        source_ = source_.at[:, :, 0].set(self.lattice.w[0] * (3 * (cx[0] * fx + cy[0] * fy) - 3 * (ux * fx + uy * fy) +
                                                    9 * (cx[0] * cx[0] * ux * fx + cy[0] * cx[0] * uy * fx + cx[0] * cy[
                    0] * ux * fy + cy[0] * cy[0] * uy * fy)))
        source_ = source_.at[:, :, 1].set(self.lattice.w[1] * (3 * (cx[1] * fx + cy[1] * fy) - 3 * (ux * fx + uy * fy) +
                                                    9 * (cx[1] * cx[1] * ux * fx + cy[1] * cx[1] * uy * fx + cx[1] * cy[
                    1] * ux * fy + cy[1] * cy[1] * uy * fy)))
        source_ = source_.at[:, :, 2].set(self.lattice.w[2] * (3 * (cx[2] * fx + cy[2] * fy) - 3 * (ux * fx + uy * fy) +
                                                    9 * (cx[2] * cx[2] * ux * fx + cy[2] * cx[2] * uy * fx + cx[2] * cy[
                    2] * ux * fy + cy[2] * cy[2] * uy * fy)))
        source_ = source_.at[:, :, 3].set(self.lattice.w[3] * (3 * (cx[3] * fx + cy[3] * fy) - 3 * (ux * fx + uy * fy) +
                                                    9 * (cx[3] * cx[3] * ux * fx + cy[3] * cx[3] * uy * fx + cx[3] * cy[
                    3] * ux * fy + cy[3] * cy[3] * uy * fy)))
        source_ = source_.at[:, :, 4].set(self.lattice.w[4] * (3 * (cx[4] * fx + cy[4] * fy) - 3 * (ux * fx + uy * fy) +
                                                    9 * (cx[4] * cx[4] * ux * fx + cy[4] * cx[4] * uy * fx + cx[4] * cy[
                    4] * ux * fy + cy[4] * cy[4] * uy * fy)))
        source_ = source_.at[:, :, 5].set(self.lattice.w[5] * (3 * (cx[5] * fx + cy[5] * fy) - 3 * (ux * fx + uy * fy) +
                                                    9 * (cx[5] * cx[5] * ux * fx + cy[5] * cx[5] * uy * fx + cx[5] * cy[
                    5] * ux * fy + cy[5] * cy[5] * uy * fy)))
        source_ = source_.at[:, :, 6].set(self.lattice.w[6] * (3 * (cx[6] * fx + cy[6] * fy) - 3 * (ux * fx + uy * fy) +
                                                    9 * (cx[6] * cx[6] * ux * fx + cy[6] * cx[6] * uy * fx + cx[6] * cy[
                    6] * ux * fy + cy[6] * cy[6] * uy * fy)))
        source_ = source_.at[:, :, 7].set(self.lattice.w[7] * (3 * (cx[7] * fx + cy[7] * fy) - 3 * (ux * fx + uy * fy) +
                                                    9 * (cx[7] * cx[7] * ux * fx + cy[7] * cx[7] * uy * fx + cx[7] * cy[
                    7] * ux * fy + cy[7] * cy[7] * uy * fy)))
        source_ = source_.at[:, :, 8].set(self.lattice.w[8] * (3 * (cx[8] * fx + cy[8] * fy) - 3 * (ux * fx + uy * fy) +
                                                    9 * (cx[8] * cx[8] * ux * fx + cy[8] * cx[8] * uy * fx + cx[8] * cy[
                    8] * ux * fy + cy[8] * cy[8] * uy * fy)))
        return source_

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

    # alternative streaming function
    # def stream(self, f):
    #     shifted_f = jnp.zeros_like(f)
    #     for k in range(self.lattice.c.shape[1]):
    #         shifted_f = shifted_f.at[:, :, k].set(jnp.roll(f[:, :, k], shift=(self.lattice.c[0, k], self.lattice.c[1, k]), axis=(0, 1)))
    #     return shifted_f

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

