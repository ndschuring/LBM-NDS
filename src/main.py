import jax
import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt
import os
import datetime
from functools import partial
import vtk
import numpy as np
from numpy.ma.core import zeros_like


# jax.config.update("jax_enable_x64", True)

class LBM:
    def __init__(self, **kwargs):
        # Initialisation Parameters
        self.nx = kwargs.get("nx") #x dimension
        self.ny = kwargs.get("ny") #y dimension
        self.nz = kwargs.get("nz") #z dimension
        self.lattice = kwargs.get("lattice") #set lattice
        # set dimensions based on lattice
        self.dimensions = [self.nx or 0, self.ny or 0, self.nz or 0]
        self.dimensions = self.dimensions[:self.lattice.d]
        self.rho_dimension = tuple(self.dimensions)
        self.u_dimension = tuple((*self.dimensions, self.lattice.d))
        # density for initialisation
        self.rho0 = kwargs.get("rho0", 1) # scalar rho0
        self.rho0_ones = jnp.ones(self.rho_dimension)*self.rho0 # matrix rho0
        # Gravity Parameters
        self.g_set = kwargs.get("g_set", 0) # gravitational constant
        self.tilt_angle = kwargs.get("tilt_angle", 0) # angle of system if gravity
        # Plotting Parameters
        self.x = jnp.arange(1, self.nx+1) - 0.5
        self.y = jnp.arange(1, self.ny+1) - 0.5
        self.plot_every = kwargs.get("plot_every", 50)
        self.plot_from = kwargs.get("plot_from", 0)
        self.sim_name = str(self)
        # Path location for storing results
        today = datetime.datetime.now().strftime(f"{self.sim_name}-%Y-%m-%d_%H-%M-%S")
        cwd = os.path.abspath(__file__)
        self.sav_dir = os.path.join(os.path.dirname(cwd), "../test", today)
        if not os.path.isdir(self.sav_dir):
            os.makedirs(self.sav_dir)
        # Boolean parameters
        self.write = kwargs.get("write", False)
        self.debug = kwargs.get("debug", False)
        self.multiphase_state = kwargs.get("multiphase_state", False)

    def __str__(self):
        """
        Fallback name if unspecified in simulation class
        :return: string representation of self
        """
        return "undefined_sim"

    def run(self, nt):
        """
        Runs the model
        1. Initialise simulation
            Initialise()
        2. iterate over nt
            update()
        3. store or plots values
            write_disk()
            plot()
        """
        f = self.initialize()
        for it in range(nt):
            f, f_prev = self.update(f) #updates f
            if it % self.plot_every == 0 and self.write:
                self.write_disk(f, nt)
            if it % self.plot_every == 0 and it >= self.plot_from:
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
        f = self.f_equilibrium(rho, u)
        return f

    @partial(jax.jit, static_argnums=0)
    def update(self, f_prev):
        """
        updates discrete velocities
            1. Calculate forcing term
                force_term()
            2. Calculate source term from force
                source_term()
            3. Collision step, apply force
                collision()
            4. Apply pre-streaming boundary conditions
                apply_pre_bc()
            5. Stream discrete velocities to neighbours
                stream()
            6. Apply post-streaming boundary conditions
                apply_bc()
        """
        force_prev = self.force_term(f_prev)
        source_prev = self.source_term(f_prev, force_prev)
        f_post_col = self.collision(f_prev, source=source_prev, force=force_prev)
        # f_post_col = self.collision(f_prev)
        f_post_col = self.apply_pre_bc(f_post_col, f_prev)
        f_post_stream = self.stream(f_post_col)
        f_post_stream = self.apply_bc(f_post_stream, f_post_col)
        return f_post_stream, f_prev

    @partial(jit, static_argnums=0, inline=True)
    def macro_vars(self, f, force=None):
        """
        Calculate macroscopic variables of f using method of moments
        0th moment: density rho
        1st moment: momentum
        Moments of g
        0th moment: phi
        1st moment: phi*u
        """
        rho = jnp.sum(f, axis=-1)
        u = jnp.dot(f, self.lattice.c.T) / rho[..., jnp.newaxis] #velocity (divide by rho)
        # u = jnp.dot(f, self.lattice.c.T)
        if force is not None:
            u += force/(2*rho[..., jnp.newaxis])
        return rho, u

    def collision(self, f,  **kwargs):
        """
        --Specified in model class--
        Applies collision operator
        """
        pass

    def apply_bc(self, f, f_prev, **kwargs):
        """
        --Specified in simulation class--
        Applies boundary conditions after streaming
        if not defined, returns post-streaming populations
        """
        return f

    def force_term(self, f, **kwargs):
        rho, u = self.macro_vars(f)
        # Gravity Force xy
        force_g = - rho * self.g_set
        force_parr = - force_g * jnp.sin(self.tilt_angle)
        force_perp = force_g * jnp.cos(self.tilt_angle)
        if self.lattice.d == 3:
            force_redundant = jnp.zeros_like(force_parr)
            force_components = jnp.stack((force_parr, force_perp, force_redundant), axis=-1)
        else:
            force_components = jnp.stack((force_parr, force_perp), axis=-1)
        return force_components

    def source_term(self, f, force):
        """
        Calculate the source term from the force density.
        :param f: lattice populations of shape (*dim, q)
        :param force: force term/density of shape (*dim, d)
        :return source_term: matrix of shape (*dim, q)
        """
        rho, u = self.macro_vars(f, force)
        cc = jnp.einsum("iq,jq->ijq", self.lattice.c, self.lattice.c)
        cc_diff = cc - (self.lattice.cs2 * jnp.eye(self.lattice.d)[...,jnp.newaxis])
        term1 = self.lattice.c/self.lattice.cs2
        term2 = jnp.einsum("abq,...b->...aq", cc_diff, u) / (self.lattice.cs2 ** 2)
        source_term = self.lattice.w*(term1 + term2)
        source_term = jnp.einsum("...ab,...a->...b", source_term, force)
        return source_term

    def apply_pre_bc(self, f, f_prev, **kwargs):
        """
        --Specified in simulation class--
        Applies boundary conditions before streaming
        if not defined, returns post-collision populations
        """
        return f

    @partial(jit, static_argnums=(0,))
    def stream(self, f):
        """
        Streams discrete velocities to neighbours in up to 3D
        Applies inherent periodic BC from rolling matrix
        """
        def stream_i(f_i, c):
            if self.lattice.d == 1:
                return jnp.roll(f_i, (c[0]), axis=0)
            if self.lattice.d == 2:
                return jnp.roll(f_i, (c[0], c[1]), axis=(0, 1))
            if self.lattice.d == 3:
                return jnp.roll(f_i, (c[0], c[1], c[2]), axis=(0, 1, 2))
        return jax.vmap(stream_i, in_axes=(-1, 0), out_axes=-1)(f, self.lattice.c.T)

    def f_equilibrium(self, rho, u, **kwargs):
        """
        Calculates the equilibrium distribution of f for a single phase
        According to equation 3.54 of the LBM book (Krüger et al.)
        :param rho: Density, shape: (*dim)
        :param u: Velocity, shape: (*dim, d)
        :param kwargs: optional arguments: None
        :return: equilibrium distribution f_eq, shape: (*dim, q)
        """
        # definitive version of equation 3.54 of LBM book. Utilizing jnp.einsum to actually understand what is going on.
        wi_rho = jnp.einsum("i,...->...i", self.lattice.w, rho)
        cc = jnp.einsum("iq,jq->ijq", self.lattice.c, self.lattice.c)
        cc_diff = cc - (self.lattice.cs2 * jnp.eye(self.lattice.d)[...,jnp.newaxis])
        uc = jnp.einsum("...j,ji->...i", u, self.lattice.c)
        uu = jnp.einsum("...a,...b->...ab", u, u)
        term1 = 1
        term2 = uc/self.lattice.cs2
        term3 = jnp.einsum("...ab,abq->...q",uu, cc_diff) / (2*self.lattice.cs2**2)
        f_eq = wi_rho * (term1 + term2 + term3)
        # f_eq = f_eq.at[..., 0].set(rho - jnp.sum(f_eq[..., 1:], axis=-1)) #correction term to ensure mass conservation?
        return f_eq

    def g_equilibrium(self, phi, u, **kwargs):
        """
        --Specified in model class--
        Calculates the equilibrium distribution of g
        """
        pass


    def write_disk(self, f, nt, **kwargs):
        """
        store macroscopic values in array for use of outside visualisation software.
        Writing an XML of some kind for VTK to be implemented in ParaView
        Unfinished
        """
        rho, u = self.macro_vars(f)
        # def save_vtk_timestep(u, rho, nt, output_path):
        #     # Create VTK grid
        #     grid = vtk.vtkstructuredGrid()
        #     grid.SetDimensions(self.nx, self.ny, 1)

            # create points
            # # Add velocity data
            # vel_flat = np.zeros((velocity.shape[0] * velocity.shape[1], 3))
            # vel_flat[:, 0] = velocity[:, :, 0].flatten()
            # vel_flat[:, 1] = velocity[:, :, 1].flatten()
            # vtk_velocity = numpy_support.numpy_to_vtk(vel_flat)
            # vtk_velocity.SetName("velocity")
            #
            # # Add density data
            # vtk_density = numpy_support.numpy_to_vtk(density.flatten())
            # vtk_density.SetName("density")
            #
            # # Attach data to grid
            # grid.GetPointData().AddArray(vtk_velocity)
            # grid.GetPointData().AddArray(vtk_density)
            #
            # # Write to file
            # writer = vtk.vtkXMLImageDataWriter()
            # writer.SetFileName(f"{output_path}/lbm_t{timestep:04d}.vti")
            # writer.SetInputData(grid)
            # writer.Write()
        pass

    def plot(self, f, it, **kwargs):
        """
        Default plotter function, 
        Specify bespoke plotter functions in simulation class
        :param f: lattice populations of shape (*dim, q)
        :param it: iteration number
        :param kwargs: optional arguments: None
        :return: None
        """
        rho, u = self.macro_vars(f)
        u_magnitude = jnp.linalg.norm(u, axis=-1, ord=2)
        # Plot velocity (magnitude or x-component of velocity vector)
        # plt.imshow(u[:,:,0].T, cmap='viridis')
        plt.imshow(u_magnitude.T, cmap='viridis')
        plt.gca().invert_yaxis()
        plt.colorbar(label="velocity magnitude")
        plt.xlabel("x [lattice units]")
        plt.ylabel("y [lattice units]")
        plt.title("it:" + str(it) + "sum_rho:" + str(jnp.sum(rho)))
        plt.savefig(self.sav_dir + "/fig_2D_it" + str(it) + ".jpg")
        plt.clf()